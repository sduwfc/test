import numpy as np
import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer
import functools
from paddlenlp.data import DataCollatorWithPadding
from paddle.io import BatchSampler, DataLoader
from sklearn.metrics import f1_score, classification_report
from paddle.metric import Metric
from paddlenlp.utils.log import logger
import paddle.nn.functional as F

from tpu_perf.infer import SGInfer


def label2id(label_file):
    label_list = {}
    with open(label_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            l = line.strip()
            label_list[l] = i
    return label_list

def read_local_dataset(path, label_list=None, is_test=False):
    """
    Read dataset
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if is_test:
                items = line.strip().split("\t")
                sentence = "".join(items)
                yield {"sentence": sentence}
            else:
                items = line.strip().split("\t")
                if len(items) == 0:
                    continue
                elif len(items) == 1:
                    sentence = items[0]
                    labels = []
                else:
                    sentence = "".join(items[:-1])
                    label = items[-1]
                    labels = [label_list[l] for l in label.split(",")]
                yield {"sentence": sentence, "label": labels}

def preprocess_function(examples, tokenizer, max_seq_length, label_nums, is_test=False):
    result = tokenizer(text=examples["sentence"], max_seq_len=max_seq_length)
    if not is_test:
        result["labels"] = [float(1) if i in examples["label"] else float(0) for i in range(label_nums)]
    return result


class MetricReport(Metric):
    """
    F1 score for multi-label text classification task.
    """

    def __init__(self, name="MetricReport", average="micro"):
        super(MetricReport, self).__init__()
        self.average = average
        self._name = name
        self.reset()

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.y_prob = None
        self.y_true = None

    def f1_score(self, y_prob):
        """
        Compute micro f1 score and macro f1 score
        """
        threshold = 0.5
        self.y_pred = y_prob > threshold
        micro_f1_score = f1_score(y_pred=self.y_pred, y_true=self.y_true, average="micro")
        macro_f1_score = f1_score(y_pred=self.y_pred, y_true=self.y_true, average="macro")
        return micro_f1_score, macro_f1_score

    def update(self, probs, labels):
        """
        Update the probability and label
        """
        if self.y_prob is not None:
            self.y_prob = np.append(self.y_prob, probs.numpy(), axis=0)
        else:
            self.y_prob = probs.numpy()
        if self.y_true is not None:
            self.y_true = np.append(self.y_true, labels.numpy(), axis=0)
        else:
            self.y_true = labels.numpy()

    def accumulate(self):
        """
        Returns micro f1 score and macro f1 score
        """
        micro_f1_score, macro_f1_score = self.f1_score(y_prob=self.y_prob)
        return micro_f1_score, macro_f1_score

    def report(self):
        """
        Returns classification report
        """
        self.y_pred = self.y_prob > 0.5
        logger.info("classification report:\n" + classification_report(self.y_true, self.y_pred, digits=4))

    def name(self):
        """
        Returns metric name
        """
        return self._name
    
class F1score():
    def __init__(self, bmodel, devices, tokenizer_name, label_file, dev_file):
        self.model = SGInfer(bmodel_path=bmodel, devices=devices)
        self.input_info = self.model.get_input_info()
        self.batch_size = self.input_info['input_ids']['shape'][0]
        self.seq_length = self.input_info['input_ids']['shape'][1]
        self.tokenizer_name = tokenizer_name
        self.label_file = label_file
        self.dev_file = dev_file
        self.metric = MetricReport()
        self.stats = dict(Micro_F1 = 0, Macro_F1 = 0)
    
        self.label_list = label2id(label_file)
        self.dev_data_loader = self.preprocess()

    def preprocess(self):
        dev_ds = load_dataset(read_local_dataset, path=self.dev_file, label_list=self.label_list, lazy=False)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=self.seq_length, label_nums=len(self.label_list))
        dev_ds = dev_ds.map(trans_func)
        collate_fn = DataCollatorWithPadding(tokenizer)
        dev_batch_sampler = BatchSampler(dev_ds, batch_size=self.batch_size, shuffle=False)
        dev_data_loader = DataLoader(dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=collate_fn)
        return dev_data_loader
    
    def evaluate(self):
        self.metric.reset()
        for batch in self.dev_data_loader:
            labels = batch.pop("labels")

            f0 = np.array(batch['input_ids'])
            f1 = np.array(batch['token_type_ids'])
            input_ids = np.pad(f0, ((0,0), (0, max(0, self.seq_length - batch['input_ids'].shape[1]))), mode='constant', constant_values=0)
            token_type_ids = np.pad(f1, ((0,0), (0, max(0, self.seq_length - batch['token_type_ids'].shape[1]))), mode='constant', constant_values=0)
            input_ids = input_ids.astype(np.int32)
            token_type_ids = token_type_ids.astype(np.int32)

            task_id = self.model.put(input_ids, token_type_ids)
            task_id, results, valid = self.model.get()
            logits = paddle.to_tensor(results[0])
            probs = F.sigmoid(logits)
            self.metric.update(probs, labels)

        self.stats['Micro_F1'], self.stats['Macro_F1'] = self.metric.accumulate()
        self.metric.reset()
    
    def get_stats(self):
        stats_t = self.stats.copy()
        stats = {k:'{:.2%}'.format(v) for k, v in stats_t.items()}
        return stats

from tpu_perf.harness import harness
@harness('f1_score')
def harness_f1_score(tree, config, args):
    input_config = config['dataset']
    tokenizer_name = tree.expand_variables(config, input_config['tokenizer_name'])
    label_file = tree.expand_variables(config, input_config['label_file'])
    dev_file = tree.expand_variables(config, input_config['dev_file'])

    bmodel = tree.expand_variables(config, args['bmodel'])

    devices = tree.global_config['devices']

    Score = F1score(bmodel, devices, tokenizer_name, label_file, dev_file)
    Score.evaluate()
    
    return Score.get_stats()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='tpu_perf F1_score harness')
    parser.add_argument('--bmodel', type=str, help='Bmodel path')
    parser.add_argument('--devices', '-d', type=int, nargs='*', help='Devices',default=[0])
    parser.add_argument('--tokenizer_name', required=True, type=str, help='Tokenizer name')
    parser.add_argument('--label_file', required=True, type=str, help='Label file')
    parser.add_argument('--dev_file', required=True, type=str, help='Development set file')
    
    args = parser.parse_args()

    Score = F1score(args.bmodel, args.devices, args.tokenizer_name, args.label_file, args.dev_file)

    Score.evaluate()
    results = Score.get_stats()
    print(results)

if __name__ == '__main__':
    main()
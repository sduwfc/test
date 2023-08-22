from .evaluate import run

from tpu_perf.harness import harness
import os


@harness("basicvsr")
def harness_main(tree, config, args):
    
    result = run(
        tree.expand_variables(config, args["spynet"]),
        tree.expand_variables(config, args["backward"]),
        tree.expand_variables(config, args["forward"]),
        tree.expand_variables(config, args["upsample"]),
        tree.expand_variables(config, config["val_file"]),
        tree.expand_variables(config, args['lrkey']),
        tree.global_config["devices"],
        kwargs=args,
    )
        # os.path.join(config["workdir"], "eval_features.pickle"),
        # os.path.join(config["workdir"], "predictions.json"),
    return result
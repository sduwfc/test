from .run import run

from tpu_perf.harness import harness
import os


@harness("bert")
def harness_main(tree, config, args):
    val_count = tree.expand_variables(config, config["val_count"])
    
    result = run(
        tree.expand_variables(config, args["bmodel"]),
        tree.expand_variables(config, config["val_file"]),
        val_count,
        os.path.join(config["workdir"], "eval_features.pickle"),
        os.path.join(config["workdir"], "predictions.json"),
        tree.global_config["devices"],
        kwargs=args,
    )
    result["exact_match"] /= val_count
    result["f1"] /= 100
    return result

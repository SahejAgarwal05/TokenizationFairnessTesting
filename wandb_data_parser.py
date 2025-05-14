import wandb
import task_configs
import pandas as pd
arc_acc_metric_keys = [ m + "/acc" for m in task_configs.arc]
arc_stderr_metric_keys = [ m + "/acc_stderr" for m in task_configs.arc]
mmlu_acc_metric_keys = [ m + "/acc" for m in task_configs.global_mmlu_lite]
mmlu_stderr_metric_keys = [ m + "/acc_stderr" for m in task_configs.global_mmlu_lite]
mgsm_lang = ["zh","th","te","sw","ru","ja","fr","es","de"]
mgsm_acc_metric_keys = [ "mgsm_native_cot_"+m + "/exact_match,flexible-extract" for m in mgsm_lang]
mgsm_stderr_metric_keys = [ "mgsm_native_cot_" +m + "/exact_match,strict-extract" for m in mgsm_lang]


arc_metirc_keys = arc_acc_metric_keys + arc_stderr_metric_keys
mgsm_metric_keys = mgsm_acc_metric_keys + mgsm_stderr_metric_keys
mmlu_metric_keys = mmlu_acc_metric_keys + mmlu_stderr_metric_keys
metric_keys = arc_metirc_keys + mgsm_metric_keys + mmlu_metric_keys

# metric_keys = mgsm_metric_keys
api = wandb.Api()
runs = api.runs(path="barid-x-ai/TokenizationFairnessTesting")
results = dict()
for run in runs:
    task,model,nshot,compression_ratio = run.name.strip().split("-")
    summary = run.summary
    task_model_nshot = task + "-" + model + "-" + nshot
    if task_model_nshot not in results:
        results[task_model_nshot] = dict()
    if "compression_ratio" not in results[task_model_nshot]:
        results[task_model_nshot]["compression_ratio"] = [compression_ratio]
    else:
        results[task_model_nshot]["compression_ratio"].append(compression_ratio)
    for k,v in summary.items():
        if k in metric_keys:
            if k in results[task_model_nshot]:
                results[task_model_nshot][k].append(v)
            else:
                results[task_model_nshot][k] = [v]
for k,v in results.items():
    try:
        df = pd.DataFrame.from_dict(v)
    except Exception:
        print(v)
    df.to_csv("./results/"+k+".csv")
import pandas as pd
import re
import glob
import os

# folder_path = './results/results/global_mmlu_lite-gemma@aya_expanse_8b-5_shot.csv'
# csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
# acc_regx = r'.+/acc$'
# stderr_regx = r'.+/acc_stderr$'
# lang_regx = r'.+_[a-z]{2}/acc_stderr$'


# dfs = []
# for file in csv_files:
#     df = pd.read_csv(file)
#     df['SourceFile'] = os.path.basename(file)
#     dfs.append(df)

# all_data = pd.concat(dfs, ignore_index=True)

# If you have CSV as string, use pd.read_csv(StringIO(csv_string))
# Else, pd.read_csv('yourfile.csv')
high_resource_languages = ["en", "ru", "zh", "de", "fr", "es", "it", "nl", "vi"]
h_regx = "|".join(high_resource_languages)
median_resource_languages = [
    "id",
    "ar",
    "hu",
    "ro",
    "da",
    "sk",
    "uk",
    "ca",
    "sr",
    "hr",
    "hi",
]
m_regx = "|".join(median_resource_languages)
low_resource_languages = ["bn", "ta", "ne", "ml", "mr", "te", "kn"]
l_regx = "|".join(low_resource_languages)

# csv_file = "./results/global_mmlu_lite-llama3_8b-5_shot.csv"

# df = pd.read_csv(csv_file)
import json
json_obj = json.load(open("/tmp/global_mmlu_lite_gpt-4.1-nano_20250519_163945.json"))
for k, v in json_obj.items():
    json_obj[k] = [v]
json_obj["compression_ratio"] = [1.0]
df = pd.DataFrame.from_dict(json_obj)
# Identify columns with /acc and /acc_stderr
# h_acc_cols = [col for col in df.columns if re.match(".+(" + h_regx + ")/acc$", col)]
# m_acc_cols = [col for col in df.columns if re.match(".+(" + m_regx + ")/acc$", col)]
# l_acc_cols = [col for col in df.columns if re.match(".+(" + l_regx + ")/acc$", col)]
h_acc_cols = [col for col in df.columns if re.match("(" + h_regx + ")$", col)]
m_acc_cols = [col for col in df.columns if re.match("(" + m_regx + ")$", col)]
l_acc_cols = [col for col in df.columns if re.match("(" + l_regx + ")$", col)]
# h_acc_cols = [col for col in df.columns if re.match(".+(" + h_regx + ")", col)]
# m_acc_cols = [col for col in df.columns if re.match(".+(" + m_regx + ")", col)]
# l_acc_cols = [col for col in df.columns if re.match(".+(" + l_regx + ")", col)]
# h_stderr_cols = [
#     col for col in df.columns if re.match(".+(" + h_regx + ")/acc_stderr$", col)
# ]
# m_stderr_cols = [
#     col for col in df.columns if re.match(".+(" + m_regx + ")/acc_stderr$", col)
# ]
# l_stderr_cols = [
#     col for col in df.columns if re.match(".+(" + l_regx + ")/acc_stderr$", col)
# ]

# df["H stderr"] = round(df[h_stderr_cols].mean(axis=1), 3)
# df["M stderr"] = round(df[m_stderr_cols].mean(axis=1), 3)
# df["L stderr"] = round(df[l_stderr_cols].mean(axis=1), 3)
df["H Acc"] = round(df[h_acc_cols].mean(axis=1), 3) * 100
df["M Acc"] = round(df[m_acc_cols].mean(axis=1), 3) * 100
df["L Acc"] = round(df[l_acc_cols].mean(axis=1), 3) * 100

# latex_table = df[['compression_ratio', 'H Acc', 'H stderr','M Acc', 'M stderr', 'L Acc', 'L stderr']].to_latex(index=False,float_format="{:.3f}".format)
latex_table = df[["compression_ratio", "H Acc", "M Acc", "L Acc"]].to_latex(
    index=False, float_format="{:.1f}".format
)
print(latex_table)

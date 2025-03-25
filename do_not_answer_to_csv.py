from glob import glob
import os
import pandas as pd

# do-not-answer-120-jaの結果ファイルを取得
model_result_paths = glob("./data/judgements/*/kunishou__do-not-answer-120-ja/*.json")

# データセットの名前を定義
eval_dataset_dict = {
    "kunishou__do-not-answer-120-ja": "Do-Not-Answer-120-ja",
}

# 結果を収集
all_result_dfs = []

for model_result_path in model_result_paths:
    temp_df = pd.read_json(model_result_path, lines=True)
    temp_df["judge_model"] = model_result_path.split("/")[3]
    temp_df["eval_dataset"] = eval_dataset_dict[model_result_path.split("/")[4]]
    temp_df["model_name"] = model_result_path.split("/")[5].replace(".json", "")
    all_result_dfs.append(temp_df)

# 結果を結合
all_result_df = pd.concat(all_result_dfs, ignore_index=True)

# データセットとカテゴリの組み合わせを作成
all_result_df["dataset_category"] = all_result_df["eval_dataset"]

# types_of_harmのカテゴリリスト（仮定、前回の5つを使用）
harm_categories = all_result_df["types_of_harm"].unique().tolist()

# モデルごとの全体平均スコアを計算
eval_dataset_names = all_result_df.eval_dataset.unique()
model_names = all_result_df.model_name.unique()

eval_corr_results = {}
for eval_dataset_name in eval_dataset_names:
    eval_corr_results[eval_dataset_name] = {}
    for model_name in model_names:
        eval_corr_results[eval_dataset_name][model_name] = all_result_df[
            (all_result_df.eval_dataset == eval_dataset_name) & 
            (all_result_df.model_name == model_name)
        ].score.mean()

# types_of_harmごとの平均スコアを計算
harm_results = {}
for harm in harm_categories:
    harm_results[harm] = {}
    for model_name in model_names:
        harm_results[harm][model_name] = all_result_df[
            (all_result_df["types_of_harm"] == harm) & 
            (all_result_df.model_name == model_name)
        ].score.mean()

# DataFrameに変換（全体スコア）
eval_res_df = pd.DataFrame(eval_corr_results)*2

# DataFrameにtypes_of_harmごとのスコアを追加
for harm in harm_categories:
    eval_res_df[harm] = pd.Series(harm_results[harm])*2

# 平均スコアを計算（モデルごとの全体平均）
eval_res_df['mean'] = eval_res_df[eval_dataset_names].mean(axis=1)*2

# スコアでソート
eval_res_df = eval_res_df.sort_values(by='mean', ascending=False)

# resultsディレクトリを作成
if not os.path.exists("results"):
    os.makedirs("results")

# CSVに保存
with open("results/do_not_answer_totals.csv", mode="w", encoding="cp932", errors="ignore", newline="") as f:
    eval_res_df.to_csv(f, index=True)

print("Do-Not-Answer-120-jaの集計結果を 'results/do_not_answer_totals.csv' に保存しました。")
print("\nモデルごとの全体平均スコアとtypes_of_harmごとの平均スコア:")
print(eval_res_df)
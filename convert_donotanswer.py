from datasets import load_dataset, Dataset
from huggingface_hub import login

# HuggingFaceにログイン（トークンが必要）
#login()

# データセットをロード
dataset = load_dataset("kunishou/do-not-answer-ja")

# 対象とするtypes_of_harmのリスト
target_harms = [
    "Adult Content",
    "Assisting illegal activities",
    "Toxic language (hate speech)",
    "Causing material harm by disseminating misinformation e.g. in medicine or law",
    "Nudging or advising users to perform unethical or unsafe actions"
]

# 各カテゴリからデータを抽出
filtered_data = []
category_data = {}  # カテゴリごとのデータを保持

for harm in target_harms:
    # 指定されたtypes_of_harmに一致するデータをフィルタリング
    harm_data = [item for item in dataset["train"] if item["types_of_harm"] == harm]
    category_data[harm] = harm_data
    # 基本の30問を取得（データが30未満の場合は全件）
    selected_data = harm_data[:min(30, len(harm_data))]
    filtered_data.extend(selected_data)

# 現在のデータ数を確認
current_count = len(filtered_data)
print(f"初期抽出データ数: {current_count}")

# 目標の150に足りない場合、他のカテゴリから補充
target_count = 150
if current_count < target_count:
    shortfall = target_count - current_count  # 不足分（例: 2）
    additional_data = []
    
    # "Adult Content"以外のカテゴリから追加
    other_categories = [h for h in target_harms if h != "Adult Content"]
    for harm in other_categories:
        available_data = category_data[harm]
        current_selected = min(30, len(available_data))  # すでに選んだ数
        if len(available_data) > current_selected and shortfall > 0:
            # 追加で1問（または不足分が残っている限り）取得
            extra_count = min(1, shortfall)  # 各カテゴリから最大1問追加
            additional_data.extend(available_data[current_selected:current_selected + extra_count])
            shortfall -= extra_count
    
    filtered_data.extend(additional_data)
    print(f"補充後のデータ数: {len(filtered_data)}")

# 新しいデータセットを作成
new_dataset = Dataset.from_list(filtered_data)

# HuggingFaceにアップロード
new_dataset.push_to_hub("umiyuki/do-not-answer-ja-creative-150", private=False)

print("データセットのアップロードが完了しました！")
print("アップロード先: https://huggingface.co/datasets/umiyuki/do-not-answer-ja-creative-150")
import argparse
import os
import logging

from datasets import load_dataset

from evaluation_datasets_config import EVAL_MODEL_CONFIGS, get_ans_path

# ロギングの設定を関数化
def setup_logging(model_name: str):
    logger = logging.getLogger()  # デフォルトのロガーを取得
    logger.setLevel(logging.INFO)  # ログレベルを設定
    
    # 既存のハンドラをクリア（重複防止）
    if logger.handlers:
        logger.handlers.clear()
    
    # フォーマットの設定
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    
    # ファイルハンドラ（model_nameを含む）
    file_handler = logging.FileHandler(f"judgement_log_{model_name.replace('/', '__')}.txt", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def evaluate(model_name: str, eval_dataset_name: str, evaluation_model: str, num_proc: int):
    
    model_answer_path = get_ans_path(eval_dataset_name, model_name)
    ans_dataset = load_dataset('json', data_files=model_answer_path, split="train")
    
    eval_config = EVAL_MODEL_CONFIGS.get(eval_dataset_name, None)
    
    if eval_config is None:
        raise ValueError(f'モデル名「{eval_dataset_name}」は対応しておりません。引数の"--eval_dataset_name"は{list(EVAL_MODEL_CONFIGS.keys())}から選択してください。')

    eval_fn = eval_config["evaluator_function"]

    ans_dataset = ans_dataset.map(lambda x: {"score": eval_fn(x, evaluation_model)}, num_proc=num_proc)
    
    ans_dataset.to_json(os.path.join(".", "data", "judgements", "judge_" + evaluation_model.replace("/", "__"), eval_dataset_name.replace("/", "__"), model_name.replace("/", "__") + ".json"), force_ascii=False)

    
def run_judgement(model_name: str, eval_dataset_name: str = "all", evaluation_model: str = "gpt-4-turbo-preview", num_proc: int = 8):
    eval_dataset_names = EVAL_MODEL_CONFIGS.keys() if eval_dataset_name == "all" else [eval_dataset_name]
    
    logger = logging.getLogger()  # 既存のロガーを取得
    for eval_dataset_name in eval_dataset_names:
        logger.info(f"Judging {model_name} on {eval_dataset_name} using {evaluation_model} ({num_proc} proc)")
        evaluate(model_name, eval_dataset_name, evaluation_model, num_proc)
        
def main():
    parser = argparse.ArgumentParser(description='Judge model answers with LLM as judge')

    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('-d', '--eval_dataset_name', type=str, default='all')
    parser.add_argument('-e', '--evaluation_model', type=str, default='gpt-4-turbo-preview')
    parser.add_argument('-n', '--num_proc', type=int, default=8)

    args = parser.parse_args()

    # 引数が確定した後にロギングを設定
    setup_logging(args.model_name)
    
    run_judgement(args.model_name, args.eval_dataset_name, args.evaluation_model, args.num_proc)
    
if __name__ == '__main__':
    main()
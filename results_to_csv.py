#!/usr/bin/env python
# coding: utf-8

# In[158]:


from glob import glob

model_result_paths = glob("./data/judgements/*/*/*.json")


# In[159]:


eval_dataset_dict = {
    "elyza__ELYZA-tasks-100": "ELYZA-tasks-100",
    "yuzuai__rakuda-questions": "Rakuda",
    "lightblue__tengu_bench": "Tengu-Bench",
    "shisa-ai__ja-mt-bench-1shot": "MT-Bench",
}


# In[160]:


import pandas as pd

all_result_dfs = []

for model_result_path in model_result_paths:
    temp_df = pd.read_json(model_result_path, lines=True)
    temp_df["judge_model"] = model_result_path.split("/")[3]
    temp_df["eval_dataset"] = eval_dataset_dict[model_result_path.split("/")[4]]
    temp_df["model_name"] = model_result_path.split("/")[5].replace(".json", "")
    
    all_result_dfs.append(temp_df)


# In[161]:


import pandas as pd

all_result_df = pd.concat(all_result_dfs)

all_result_df["dataset_category"] = all_result_df["eval_dataset"] + " " + all_result_df["Category"]

# In[162]:


eval_dataset_names = all_result_df.eval_dataset.unique()
model_names = all_result_df.model_name.unique()


# In[163]:


eval_corr_results = {}
for eval_dataset_name in eval_dataset_names:
    eval_corr_results[eval_dataset_name] = {}
    for model_name in model_names:
        eval_corr_results[eval_dataset_name][model_name] = all_result_df[(all_result_df.eval_dataset == eval_dataset_name) & (all_result_df.model_name == model_name)].score.mean()


# In[164]:


pd.DataFrame(eval_corr_results).corr().round(4)


# In[166]:


eval_res_df = pd.DataFrame(eval_corr_results)

eval_res_df['ELYZA-tasks-100'] = eval_res_df['ELYZA-tasks-100'] * 2

eval_res_df['mean'] = eval_res_df.mean(axis=1)

eval_res_df = eval_res_df.sort_values(by='mean', ascending=False)

def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

styled_df = eval_res_df.style.apply(highlight_max, axis=0)
styled_df = styled_df.format("{:.2f}")
styled_df
eval_res_df.to_csv("to_csv_out.csv")
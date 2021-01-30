import pandas as pd
import json
# 标注数据
df = pd.read_csv('./data/poem_shi.csv')
df_train = df[:-6000]
df_dev = df[-6000:]

df_dict = dict(zip(df_train['input'], df_train['output']))
poem_data = []
for i, j in df_dict.items():
    k = i.split('&&')[-1]
    ii = i.split('&&')[:-1][0]
    instance = {'src_text': ii, 'tgt_text': j, 'ctrl_text': k}
    poem_data.append(instance)
with open('data/train_data.json', "w", encoding='utf-8') as result:
    for each in poem_data:
        result.write(json.dumps(each, ensure_ascii=False) + '\n')

df_dict = dict(zip(df_dev['input'], df_dev['output']))
poem_data = []
for i, j in df_dict.items():
    k = i.split('&&')[-1]
    ii = i.split('&&')[:-1]
    instance = {'src_text': ii[0], 'tgt_text': j, 'ctrl_text': k}
    poem_data.append(instance)
with open('data/dev_data.json', "w", encoding='utf-8') as result:
    for each in poem_data:
        result.write(json.dumps(each, ensure_ascii=False) + '\n')


import os
import pandas as pd
import csv
import re
df = pd.read_csv("F:\PhD work\Idioms\idiom_detection_nlp\preprocess_data\dataset_out.csv")

# print(df.head())
def getsubidx(x, y):
    l1, l2 = len(x), len(y)
    for i in range(l1):
        if x[i:i+l2] == y:
            return [i,i+l2]
tags_list = []
for _, row in df.iterrows():

    sent_tokens = row['sentence'].replace(',','').replace(';','').replace('?','').replace("'",'').replace('.','').split(" ")
    tags = [0] * len(sent_tokens)
    idiom_tokens = row['idiomatic_part'].replace(',','').replace(';','').replace('?','').replace("'",'').replace('.','').split(" ")
    print(sent_tokens, idiom_tokens)
    period = getsubidx(sent_tokens,idiom_tokens)
    tags[period[0]:period[1]] = [1]*(period[1]-period[0])
    print(tags)
    tags_list.append(tags)


df['tags'] = tags_list

df.to_csv("with_tags.csv")
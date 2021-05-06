import pandas as pd
import ast
dataset = pd.read_csv("E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\Idiom_scraper\with_tags_theidioms.com.csv")
print(dataset.columns)
tokens = []
labels = []
for ind,k in dataset.iterrows():
    sent_tokens = ast.literal_eval(k['sentence_tokens'])
    idi_tokens = ast.literal_eval(k['tags'])
    # print(idi_tokens)
    for st in sent_tokens:
        tokens.append(st)
    labels.extend(idi_tokens)


out_df = pd.DataFrame()
out_df['tokens'] = tokens
out_df['labels'] = labels

out_df.to_csv('processed_data_bow.csv')

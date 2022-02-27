import pandas as pd


emo_data = pd.read_csv("E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\\building_emotional_embeddings\data\sentences.csv")
emo_data.columns = ['#','sentence','sentiment']
sentiment_ids = []

for i,row in emo_data.iterrows():
    if(row['sentiment']=='positive'):sentiment_ids.append(1)
    elif(row['sentiment']=='negative'):sentiment_ids.append(0)
    elif (row['sentiment'] == 'other'): sentiment_ids.append(2)

emo_data['sentiment_id'] = sentiment_ids
emo_data.to_csv("E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\\building_emotional_embeddings\data\sentences_wc.csv")


import pandas as pd
import json
import csv
data_path = "E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\\building_emotional_embeddings\EmotionLines\Friends\\friends_train.json"

# with open(data_path) as jsonfile:
#     # data = json.load(jsonfile)
#     # print(len(data[0]))
#     # print(len(data[1]))
#     # print(data[0])
#     # print(data[1])
#     # f = csv.writer(open("E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\\building_emotional_embeddings\EmotionLines\procesd_emotionlines.csv", "w", newline='',encoding='utf-8'))
#     #
#     # # Write CSV Header, If you dont need that, remove this line
#     # f.writerow(["speaker", "utterance", "emotion"])
#     #
#     # for sc in data:
#     #     for each_s in sc:
#     #         print(each_s)
#     #     # print(x)
#     #         f.writerow([each_s["speaker"],
#     #                     each_s["utterance"],
#     #                     each_s["emotion"]])
#     #     # break
#     #
#     # # print(df.head())


df = pd.read_csv('E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\\building_emotional_embeddings\EmotionLines\procesd_emotionlines.csv')
emo_dict = {'neutral':0, 'surprise':1, 'fear':2, 'non-neutral':3, 'joy':4, 'sadness':5, 'anger':6,'disgust':7}
print(df['emotion'].unique())
emo_ids = []
for i,row in df.iterrows():
    emotion_id = emo_dict[row['emotion']]
    emo_ids.append(emotion_id)

df['emotion_ids'] = emo_ids

df.to_csv('E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\\building_emotional_embeddings\EmotionLines\procesd_emotionlines_emo_ids.csv')

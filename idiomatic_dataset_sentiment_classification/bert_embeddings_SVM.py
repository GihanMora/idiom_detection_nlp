from transformers import AutoTokenizer, AutoModel
import torch
import numpy
import time

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support, \
    confusion_matrix
import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

data = pd.read_csv("E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\\building_emotional_embeddings\data\sentences_wc.csv")

sentences = list(data["sentence"])
labels = list(data["sentiment_id"])
#Sentences we want sentence embeddings for
# sentences = ['This framework generates embeddings for each input sentence',
#              'Sentences are passed as a list of string.',
#              'The quick brown fox jumps over the lazy dog.']
# model_name = "ProsusAI/finbert"
model_name_base = "bert-base-uncased"
# model_name = "E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\\building_emotional_embeddings\models\emotion_lines_500_steps\checkpoint-1500\\"
model_name = "E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\models\\epie_models"
#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained(model_name_base)
model = AutoModel.from_pretrained(model_name_base)

#Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=32, return_tensors='pt')

#Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

#Perform pooling. In this case, mean pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
sentence_embeddings = sentence_embeddings.tolist()

print(numpy.shape(sentence_embeddings))
X_train, X_val, y_train, y_val = train_test_split(sentence_embeddings, labels, test_size=0.2,random_state=42)




start = time.time()

svm_classifier = SVC()
svm_classifier.fit(X_train,y_train)

end = time.time()
process = round(end-start,2)
print("Support Vector Machine Classifier has fitted, this process took {} seconds".format(process))

# print(svm_classifier.score(X_val,y_val))

predicted_y = svm_classifier.predict(X_val)
print(predicted_y)

def compute_metrics(pred,ground_labels):
    labels_all = ground_labels
    preds_all = list(pred)


    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all)
    acc = accuracy_score(labels_all, preds_all)
    confusion_mat = confusion_matrix(labels_all, preds_all)

    out_dict = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusiton_mat': confusion_mat
    }
    for k in out_dict.keys():
        print(k)
        print(out_dict[k])

compute_metrics(predicted_y,y_val)



#fin
# tensor([[ 0.0102,  0.8853, -0.2012,  ..., -0.6074, -0.5253,  0.0979],
#         [ 0.2248,  0.5273, -0.0640,  ..., -0.1256, -0.5626,  0.1556],
#         [ 0.0105,  0.1267,  0.0584,  ..., -0.2656,  0.5670,  0.4216]])

#base
# tensor([[-0.1256, -0.0235,  0.0972,  ..., -0.1809, -0.3674,  0.2712],
#         [ 0.0789, -0.2891,  0.0363,  ..., -0.1744, -0.4208,  0.6002],
#         [-0.0145, -0.0749,  0.0564,  ..., -0.2625,  0.4954,  0.0740]])


# just_bert
# accuracy
# 0.6607142857142857
# f1
# [0.75721562 0.63157895 0.37209302]
# precision
# [0.6408046  0.76470588 0.59259259]
# recall
# [0.9253112  0.53793103 0.27118644]
# confusiton_mat
# [[223   8  10]
#  [ 55  78  12]
#  [ 70  16  32]]


#emo bert 2500
# accuracy
# 0.6111111111111112
# f1
# [0.73010381 0.58730159 0.25842697]
# precision
# [0.63173653 0.60162602 0.4893617 ]
# recall
# [0.8647541  0.57364341 0.17557252]
# confusiton_mat
# [[211  20  13]
#  [ 44  74  11]
#  [ 79  29  23]]


# bert_emo_500
# accuracy
# 0.6686507936507936
# f1
# [0.74637681 0.6870229  0.42268041]
# precision
# [0.65605096 0.72       0.63076923]
# recall
# [0.86554622 0.65693431 0.31782946]
# confusiton_mat
# [[206  17  15]
#  [ 38  90   9]
#  [ 70  18  41]]


# bert emo 1500

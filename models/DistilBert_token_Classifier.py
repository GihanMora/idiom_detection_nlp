import pandas as pd
import ast
from sklearn.model_selection import train_test_split
df = pd.read_csv("F:\PhD work\Idioms\idiom_detection_nlp\preprocess_data\with_tags.csv")

token_docs = []
tag_docs = []


for _, row in df.iterrows():
    token_docs.append(ast.literal_eval(row['sentence_tokens']))
    tag_docs.append(ast.literal_eval(row['tags']))


# print(token_docs)

train_texts, val_texts, train_tags, val_tags = train_test_split(token_docs, tag_docs, test_size=.2)
# print(train_texts)
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)



import numpy as np

def encode_tags(tags, encodings):
    labels = tags
    encoded_labels = []
    i=0
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # print(len(doc_labels),len(doc_offset))

        try:
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

            # print(doc_enc_labels.tolist())
            # break
        except Exception as e:

            print()
            print(e)
            print(token_docs[i])
            print(tag_docs[i])
            # print(len(doc_labels), len(doc_offset))
            len1 = len(doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)])
            len2 = len(doc_labels)
            print(len1,len2)
            if(len1<len2):
                doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels[:len1]
            elif (len1 > len2):
                doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)][:len2] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())
            # break

        i=i+1
    return encoded_labels

train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)


import tensorflow as tf

train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
))

from transformers import TFDistilBertForTokenClassification
model = TFDistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=2)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss) # can also use any keras loss fn
model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16)


preditions = model.predict(val_dataset)

results = pd.DataFrame()
results['ground'] = val_tags
results['predictions'] = preditions
results.to_csv('output_results.csv')
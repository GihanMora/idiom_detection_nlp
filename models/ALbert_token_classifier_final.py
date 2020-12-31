import pandas as pd
import ast
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,confusion_matrix
from transformers import DistilBertTokenizerFast, XLNetTokenizerFast, TFXLNetForTokenClassification, AlbertTokenizer, \
    AlbertForTokenClassification
import tensorflow as tf
from transformers import TFDistilBertForTokenClassification, TFTrainer, TFTrainingArguments
from transformers import EvaluationStrategy
import numpy as np


#Building a prediction dataset with dummy labels
def make_test_data(sentences):
    print('making test data')
    w_encodings = tokenizer(sentences, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                              truncation=True, max_length=30)
    w_encodings.pop("offset_mapping")
    out_dataset = tf.data.Dataset.from_tensor_slices((
        dict(w_encodings)
        , [[0] * 30] * len(w_encodings['input_ids'])
        ,  # val_labels
    ))

    return out_dataset

#converting tokenizer embedded token id lists to human readable word lists
def convet_encordings_to_words(input_encodings):
    tokens = []
    for each_sent in input_encodings["input_ids"]:
        tokens.append(tokenizer.convert_ids_to_tokens(each_sent))
    return tokens

#reading dataset given the csv path
def read_dataset(path):
    df = pd.read_csv(path)
    token_docs = []
    tag_docs = []

    for _, row in df.iterrows():
        token_docs.append(ast.literal_eval(row['sentence_tokens']))
        tag_docs.append(ast.literal_eval(row['tags']))
    return [token_docs,tag_docs]

#reencode the labels such that it preserves the token-label alignment
def encode_tags(tags, encodings):
    labels = tags
    encoded_labels = []
    i=0
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):

        try:
            # create an empty array of -100
            # doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
            doc_enc_labels = np.zeros(len(doc_offset), dtype=int)
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

        except Exception as e:

            len1 = len(doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)])
            len2 = len(doc_labels)
            # print(len1,len2)
            if(len1<len2):
                doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels[:len1]
            elif (len1 > len2):
                doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)][:len2] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())
            # break

        i=i+1
    return encoded_labels

#computer matrices of prediction evaluations
def compute_metrics(pred,ground_labels):
    labels = ground_labels
    preds = pred.predictions.argmax(-1)
    labels_all = []
    preds_all = []
    for l in labels:
        labels_all=labels_all+l

    for p in preds:
        preds_all=preds_all+list(p)

    labels_all = labels_all
    preds_all = preds_all[:len(labels_all)]

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


data_read = read_dataset("..\preprocess_data\with_tags.csv")
token_docs = data_read[0]
tag_docs = data_read[1]

train_texts, val_texts, train_tags, val_tags = train_test_split(token_docs, tag_docs, test_size=.2, random_state=42)

#setting the tokenizer
tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')


train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,truncation=True, max_length=30)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=30)


train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)

train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")

#building tensor datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings)
    # ,[[0]*30]*len(val_encodings['input_ids'])#dummy labels
    , val_labels
))

#training arguments
training_args = TFTrainingArguments(
    output_dir='E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\models\\',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=5,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\models\\',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy= EvaluationStrategy.NO,

)

with training_args.strategy.scope():
    model = AlbertForTokenClassification.from_pretrained('albert-base-v2',num_labels=2)
    # model = TFXLNetForTokenClassification.from_pretrained('xlnet-base-cased', num_labels=2)

trainer = TFTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    # eval_dataset=val_dataset             # evaluation dataset
)


trainer.train()


#predictions
prediction_results = trainer.predict(test_dataset=test_dataset)
trainer.save_model("E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\models\\")

compute_metrics(prediction_results,val_labels)
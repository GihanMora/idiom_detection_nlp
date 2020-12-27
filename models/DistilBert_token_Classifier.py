import pandas as pd
import ast
from sklearn.model_selection import train_test_split
df = pd.read_csv("..\preprocess_data\with_tags.csv")
import tensorflow as tf
token_docs = []
tag_docs = []
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,confusion_matrix

for _, row in df.iterrows():
    token_docs.append(ast.literal_eval(row['sentence_tokens']))
    tag_docs.append(ast.literal_eval(row['tags']))


# print(token_docs)

train_texts, val_texts, train_tags, val_tags = train_test_split(token_docs, tag_docs, test_size=.2, random_state=42)
# train_texts, val_texts, train_tags, val_tags = train_test_split(token_docs, tag_docs, test_size=.2)
# print(train_texts[:3])
print([len(train_texts[0]),len(train_texts[1]),len(train_texts[2])])
# print(train_tags[:3])
print([len(train_texts[0]),len(train_texts[1]),len(train_texts[2])])
print('val texts and tags')
print(['words before tokenize',val_texts[1],len(val_texts[1])])
print(['tags before tokenize',val_tags[1],len(val_tags[1])])
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,truncation=True, max_length=30)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=30)
val_tokens = []
print(['count val encodings',len(val_encodings),val_encodings['input_ids'][:2],val_encodings['attention_mask'][:2],val_encodings['offset_mapping'][:2]])
for each_sent in val_encodings["input_ids"]:
    val_tokens.append(tokenizer.convert_ids_to_tokens(each_sent))
print("encorded texts")
print(['encorded tokens ids',val_encodings["input_ids"][1],len(val_encodings["input_ids"][1])])
print(['encorded tokens',val_tokens[1],len(val_tokens[1])])
vt = open('validation_tokens.txt','w')
for k in val_tokens:
    vt.write(str(k)+'\n')

vt.close()
# print(train_encodings["tokens"][:3])
# print(train_encodings["attention_mask"][:3])

import numpy as np

def encode_tags(tags, encodings):
    labels = tags
    encoded_labels = []
    i=0
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # print(len(doc_labels),len(doc_offset))

        try:
            # create an empty array of -100
            # doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
            doc_enc_labels = np.zeros(len(doc_offset), dtype=int)
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

            # print(doc_enc_labels.tolist())
            # break
        except Exception as e:

            # print()
            # print(e)
            # print(token_docs[i])
            # print(tag_docs[i])
            # print(len(doc_labels), len(doc_offset))
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

train_labels = encode_tags(train_tags, train_encodings)

print(['lensss',len(val_tags),len(val_encodings["input_ids"]),len(val_encodings["offset_mapping"]),len(val_encodings["attention_mask"])])
val_labels = encode_tags(val_tags, val_encodings)
print('labels')
print(['len labels',len(val_labels)])
print(['encoded_tags',val_labels[1],len(val_labels[1])])
# import sys
# sys.exit()
#without encoding tags
# train_labels = data_tensor = tf.ragged.constant(train_tags)
# val_labels = data_tensor = tf.ragged.constant(val_tags)
# data_tensor = tf.ragged.constant(data)
import tensorflow as tf

train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))

# print(len(dict(val_encodings)))
# print(len(val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings)
    ,[[0]*30]*len(val_encodings['input_ids'])
    ,# val_labels
))

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
    print('w_enc',w_encodings['input_ids'])
    print([[0] * 30] * len(w_encodings['input_ids']))
    return out_dataset





# print("for last time")
# print(len(val_dataset))
# for ele in val_dataset:
#     print(ele)



# # #Use tensorflow to train and evaluate
# from transformers import TFDistilBertForTokenClassification
# model = TFDistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=2)
# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
# model.compile(optimizer=optimizer, loss=model.compute_loss,metrics=["accuracy"]) # can also use any keras loss fn
# history = model.fit(train_dataset.shuffle(1000).batch(16), epochs=1, batch_size=16)
# # model.save("E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\models\model_files\\")
# # import tensorflow as tf
# # model = tf.keras.models.load_model("E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\models\model_files\\")
# # Evaluate the model on the test data using `evaluate`
#
# model_config = {'model':"TFDistilBertForTokenClassification",
#                 'tokenizer':"DistilBertTokenizerFast",
#                 'lr':"5e-5",
#                 'Epochs':"5",
#                 'Batch-size':"16"
#                 }
#
# out_name = model_config['model']+'_'+model_config['Epochs']+'_results.txt'
# f = open(out_name+'try',"w")
# f.write("Model Configurations\n")
# f.write('Model '+model_config['model']+'\n')
# f.write('Tokenizer '+model_config['tokenizer']+'\n')
# f.write('leaning rate '+model_config['lr']+'\n')
# f.write('Batch Size '+model_config['Batch-size']+'\n')
# f.write('Model Training History'+'\n')
# f.write('Model History '+str(history.history)+'\n')
# f.write("Evaluate on test data"+'\n')
# #validate
# # results = model.evaluate(val_dataset)
# # f.write("test loss, test acc: "+str(results)+'\n')
#
# # predict
# #getting only a part
# # print(['count val encodings',len(val_encodings),val_encodings['input_ids'][:2],val_encodings['attention_mask'][:2]])
# # dict_slice = {'input_ids':val_encodings['input_ids'][:2],'attention_mask':val_encodings['attention_mask'][:2]}
# # val_dataset_slice = tf.data.Dataset.from_tensor_slices((
# #     dict(dict_slice),
# #     val_labels[:2]
# # ))
# results1 = model.predict(val_dataset)
# # print(len(val_dataset[0]))
# print(len(results1['logits']))
# logits_list = results1['logits']
# print("Predictions")
# # pres_1d_list = [list(i)[0] for i in list(logits_list)]
# # print(pres_1d_list)
# # max_y_pred_test = np.argmax(logits_list, axis=1)
# # print(max_y_pred_test)
# pred_list = []
# # print(results1['logits'])
# for i in logits_list:
#     # print(i)
#     # print(i.argmax(-1))
#     pred_list.append(list(i.argmax(-1))[0])
# f.close()
#
# def compute_metrics_tf():
#
#     labels = val_labels
#     print('labs',labels)
#     l_all = []
#     for i in labels:
#         print(i)
#         l_all = l_all + i
#     preds = pred_list
#     print('l_all',l_all)
#     print('preds_',preds)
#
#     precision, recall, f1, _ = precision_recall_fscore_support(l_all, preds)
#     acc = accuracy_score(l_all, preds)
#     confusion_mat = confusion_matrix(l_all, preds)
#
#     out_dict = {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall,
#         'conf_labels':"horizontal-predictions, vertical labels",
#         'confusiton_mat': confusion_mat
#     }
#     for k in out_dict.keys():
#         print(k)
#         print(out_dict[k])
#
#         # break
# compute_metrics_tf()

# # # Generate predictions (probabilities -- the output of the last layer)
# # # on new data using `predict`
# # print("Generate predictions for 3 samples")
# # predictions = model.predict(val_dataset)
# # result = tf.argmax(predictions.logits).numpy()
# # print(result)
# # print("predictions shape:", predictions.shape)
# # model = loa
# # print(predictions)
# # preditions = model.predict(val_dataset[0])
# # print(len(val_tags),len(preditions))
# # results = pd.DataFrame()
# # # results['ground'] = val_tags
# # # results['predictions'] = predictions
# # results.to_csv('output_results.csv')


#Using HuggingFace trainer
from transformers import TFDistilBertForTokenClassification, TFTrainer, TFTrainingArguments
from transformers import EvaluationStrategy
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
    model = TFDistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=2)

trainer = TFTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    # eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()

# test_dataset = make_test_data([['this','is','a','moot','point'],['my','name','is','gihan']])
# a_dataset = make_test_data(val_texts[:2])
results1 = trainer.predict(test_dataset=test_dataset)
trainer.save_model("E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\models\\")
print(results1)
print("predictions>>")
print(results1.predictions[:2])
print("labels>>")
print(results1.label_ids[:2])


print("lennnnnn")
print([len(results1.predictions)])
print("mannn")
def compute_metrics(pred):

    labels = val_labels
    preds = pred.predictions.argmax(-1)
    # print(labels[1])
    # print(preds[1])
    labels_all = []
    preds_all = []
    fl = open('labels.txt','w')
    fp = open('predictions.txt', 'w')
    for l in labels:
        fl.write(str(l)+'\n')
        labels_all=labels_all+l
    for p in preds:
        print(p)
        fp.write(str(list(p)) + '\n')
        preds_all=preds_all+list(p)

    fl.close()
    fp.close()
    labels_all = labels_all[:15150]
    preds_all = preds_all[:15150]

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
    # for each_i in range(len(labels)):
    #     precision, recall, f1, _ = precision_recall_fscore_support(labels[each_i], preds[each_i])
    #     acc = accuracy_score(labels[each_i], preds[each_i])
    #     confusion_mat = confusion_matrix(labels[each_i], preds[each_i])
    #
    #     out_dict = {
    #         'accuracy': acc,
    #         'f1': f1,
    #         'precision': precision,
    #         'recall': recall,
    #         'confusiton_mat': confusion_mat
    #     }
    #     for k in out_dict.keys():
    #         print(k)
    #         print(out_dict[k])
    #
    #     break



output = compute_metrics(results1)

# print(output)
# for k in output.keys():
#     print(k)
#     print(output[k])
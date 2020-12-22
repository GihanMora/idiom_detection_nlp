import pandas as pd
import ast
from sklearn.model_selection import train_test_split
df = pd.read_csv("..\preprocess_data\with_tags.csv")
import tensorflow as tf
token_docs = []
tag_docs = []


for _, row in df.iterrows():
    token_docs.append(ast.literal_eval(row['sentence_tokens']))
    tag_docs.append(ast.literal_eval(row['tags']))


# print(token_docs)

train_texts, val_texts, train_tags, val_tags = train_test_split(token_docs, tag_docs, test_size=.2, random_state=42)
# print(train_texts[:3])
print([len(train_texts[0]),len(train_texts[1]),len(train_texts[2])])
# print(train_tags[:3])
print([len(train_texts[0]),len(train_texts[1]),len(train_texts[2])])
print('val texts and tags')
print(['words before tokenize',val_texts[1],len(val_texts[1])])
print(['tags before tokenize',val_tags[1],len(val_tags[1])])
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,truncation=True, max_length=10)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=10)
val_tokens = []
for each_sent in val_encodings["input_ids"]:
    val_tokens.append(tokenizer.convert_ids_to_tokens(each_sent))
print("encorded texts")
print(['encorded tokens ids',val_encodings["input_ids"][1],len(val_encodings["input_ids"][1])])
print(['encorded tokens',val_tokens[1],len(val_tokens[1])])

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
            doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
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
val_labels = encode_tags(val_tags, val_encodings)
print('labels')
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
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
))



# #Use tensorflow to train and evaluate
# from transformers import TFDistilBertForTokenClassification
# model = TFDistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=2)
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
# model.compile(optimizer=optimizer, loss=model.compute_loss,metrics=["accuracy"]) # can also use any keras loss fn
# history = model.fit(train_dataset.shuffle(1000).batch(16), epochs=5, batch_size=16)
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
# results = model.evaluate(val_dataset)
# f.write("test loss, test acc: "+str(results)+'\n')
# f.close()
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
    per_device_eval_batch_size=64,   # batch size for evaluation
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
results1 = trainer.predict(test_dataset=val_dataset)
trainer.save_model("E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\models\\")



print(results1)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(pred):

    labels = val_labels
    preds = pred.predictions.argmax(-1)
    print(labels[1])
    print(preds[1])
    precision, recall, f1, _ = precision_recall_fscore_support(labels[:500], preds[:500], average='binary')
    acc = accuracy_score(labels, preds)


    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


output = compute_metrics(results1)

print(output)
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from pathlib import Path
import re

def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

texts, tags = read_wnut('E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\W_NUT_dataset\wnut17train.conll\\')

token_docs = texts
tag_docs = tags





# print(token_docs)

train_texts, val_texts, train_tags, val_tags = train_test_split(token_docs, tag_docs, test_size=.2)
unique_tags = set(tag for doc in tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}
print(unique_tags)
print(id2tag)
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
import numpy as np

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -1
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)
print(train_labels[:5])
print(val_labels[:5])
#without encoding tags
# train_labels = train_tags
# val_labels = val_tags

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



#Use tensorflow to train and evaluate
from transformers import TFDistilBertForTokenClassification
model = TFDistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(unique_tags))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=model.compute_loss,metrics=["accuracy"]) # can also use any keras loss fn
history = model.fit(train_dataset.shuffle(100).batch(16), epochs=3, batch_size=16)
# model.save("E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\models\model_files\\")
# import tensorflow as tf
# model = tf.keras.models.load_model("E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\models\model_files\\")
# Evaluate the model on the test data using `evaluate`

model_config = {'model':"TFDistilBertForTokenClassification_W_NUT",
                'tokenizer':"DistilBertTokenizerFast",
                'lr':"0.01",
                'Epochs':"5",
                'Batch-size':"16"
                }

out_name = model_config['model']+'_'+model_config['Epochs']+'_results.txt'
f = open(out_name,"w")
f.write("Model Configurations\n")
f.write('Model '+model_config['model']+'\n')
f.write('Tokenizer '+model_config['tokenizer']+'\n')
f.write('leaning rate '+model_config['lr']+'\n')
f.write('Batch Size '+model_config['Batch-size']+'\n')
f.write('Model Training History'+'\n')
f.write('Model History '+str(history.history)+'\n')
f.write("Evaluate on test data"+'\n')
results = model.evaluate(val_dataset)
f.write("test loss, test acc: "+str(results)+'\n')
f.close()
# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
# print("Generate predictions for 3 samples")
# predictions = model.predict(val_dataset)
# result = tf.argmax(predictions.logits).numpy()
# print(result)
# print("predictions shape:", predictions.shape)
# model = loa
# print(predictions)
# preditions = model.predict(val_dataset[0])
# print(len(val_tags),len(preditions))
# results = pd.DataFrame()
# # results['ground'] = val_tags
# # results['predictions'] = predictions
# results.to_csv('output_results.csv')

# #Using HuggingFace trainer
# from transformers import TFDistilBertForTokenClassification, TFTrainer, TFTrainingArguments
# from transformers import EvaluationStrategy
# training_args = TFTrainingArguments(
#     output_dir='E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\models\\',          # output directory
#     num_train_epochs=3,              # total number of training epochs
#     per_device_train_batch_size=16,  # batch size per device during training
#     per_device_eval_batch_size=64,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\models\\',            # directory for storing logs
#     logging_steps=10,
#
#     evaluation_strategy= EvaluationStrategy.NO,
#
# )
#
# with training_args.strategy.scope():
#     model = TFDistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=2)
#
# trainer = TFTrainer(
#     model=model,                         # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=train_dataset,         # training dataset
#     # eval_dataset=val_dataset             # evaluation dataset
# )
#
# results = trainer.train()
# results1 = trainer.predict(test_dataset=val_dataset)
# trainer.save_model("E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\models\\")
# print(results)
#
# print(results1)

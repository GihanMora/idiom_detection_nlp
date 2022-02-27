import numpy
from torch import tensor

model_path = "E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\\building_emotional_embeddings\models\idiomatic_dataset_with_sentiments\checkpoint-500\\"
# model_path = 'bert-base-uncased'
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, BertModel, BertForSequenceClassification
import torch
# print('hi')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertModel.from_pretrained(model_path,output_hidden_states=True)
sentence = "Hello developmentation"
tokens = tokenizer.tokenize(sentence)
print(tokens)
#
# # input_ids = tf.constant(tokenizer.encode(sentence))[None, :]  # Batch size 1
input_ids = torch.tensor(tokenizer.encode(sentence))[None, :]  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs  # The last hidden-state is the first element of the output tuple
# print(last_hidden_states)

#concatenating last four hidden layers
hidden_states = outputs['hidden_states']
pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
# average_last_4_hidden_states = torch.mean(tensor([hidden_states[-1],hidden_states[-2]]))
print('this',numpy.shape(pooled_output))
# print('that',numpy.shape(average_last_4_hidden_states))
# https://github.com/huggingface/transformers/issues/1328


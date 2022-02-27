import numpy
import pandas as pd

from idiom_detection_nlp.getting_embeddings.plot_embeddings import tsne_vocab_plot

model_path = "E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\models\\epie_models"
# model_path = 'bert-base-uncased'
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, BertModel, BertForSequenceClassification
import torch
print('hi')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained(model_path,output_hidden_states=True)


def get_embeddings(sentence):
    tokens = tokenizer.tokenize(sentence)
    print(tokens)

    input_ids = tf.constant(tokenizer.encode(sentence))[None, :]  # Batch size 1

    outputs = model(input_ids)
    # last_hidden_states = outputs  # The last hidden-state is the first element of the output tuple
    # print(last_hidden_states)

    #concatenating last four hidden layers
    hidden_states = outputs['hidden_states']
    pooled_output = tf.keras.layers.concatenate(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]))
    average_last_hidden_state = tuple([hidden_states[i] for i in [ -1]])
    embeddings = average_last_hidden_state[0][0][1:-1]
    # print('this',numpy.shape(pooled_output))
    # print('that',numpy.shape(average_last_4_hidden_states))

    return {'tokens':tokens,'embeddings':embeddings}
# https://github.com/huggingface/transformers/issues/1328


sentence = "That was a moot point for them"
out_embs = get_embeddings(sentence)
tokens = out_embs['tokens']
embeddings= out_embs['embeddings']
labels = ['1']*len(tokens)

vocab_df = pd.DataFrame()
vocab_df['token'] = tokens
vocab_df['label'] = tokens
vocab_df['embedding'] = embeddings
tsne_vocab_plot(vocab_df)




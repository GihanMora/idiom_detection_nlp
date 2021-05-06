model_path = "E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\models\\epie_models"
model_path = 'bert-base-uncased'
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
print('hi')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained(model_path)
sentence = "Hello developmentation"
tokens = tokenizer.tokenize(sentence)
print(tokens)

input_ids = tf.constant(tokenizer.encode(sentence))[None, :]  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
print(last_hidden_states)
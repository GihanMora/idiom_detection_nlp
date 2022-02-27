from transformers import AutoTokenizer, AutoModel
import torch
import numpy

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask



#Sentences we want sentence embeddings for
sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']
model_name = "ProsusAI/finbert"
# model_name = "bert-base-uncased"
#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

#Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

#Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

#Perform pooling. In this case, mean pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print(sentence_embeddings)

#fin
# tensor([[ 0.0102,  0.8853, -0.2012,  ..., -0.6074, -0.5253,  0.0979],
#         [ 0.2248,  0.5273, -0.0640,  ..., -0.1256, -0.5626,  0.1556],
#         [ 0.0105,  0.1267,  0.0584,  ..., -0.2656,  0.5670,  0.4216]])

#base
# tensor([[-0.1256, -0.0235,  0.0972,  ..., -0.1809, -0.3674,  0.2712],
#         [ 0.0789, -0.2891,  0.0363,  ..., -0.1744, -0.4208,  0.6002],
#         [-0.0145, -0.0749,  0.0564,  ..., -0.2625,  0.4954,  0.0740]])
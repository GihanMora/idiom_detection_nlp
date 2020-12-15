import os

import pandas as pd
import tqdm

sentences = pd.read_csv('sentences.csv',header=None)
output_root = 'F:\\PhD work\\Idioms\\idiom_detection_nlp\\sentences_processed\\'
print(sentences.columns)
for _, row in sentences.iterrows():
  file_name = 'sentence_'+str(row[0])+'.txt'
  f = open(os.path.join(output_root,file_name),'w')
  f.write(row[1])
  f.close()
  print(row[1])

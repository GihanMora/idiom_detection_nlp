import os
import pandas as pd
import csv
f = open("F:\\PhD work\\Idioms\\idiom_detection_nlp\\tools\\annotations.txt","r")

sentence_root = "F:\PhD work\Idioms\idiom_detection_nlp\sentences_processed\\"

with open('dataset_out.csv', mode='w',  newline='') as res_file:
    res_writer = csv.writer(res_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    res_writer.writerow(['sentence_file', 'sentence', 'idiomatic_part', 'ip_start', 'ip_length'])
    for line in f.readlines():
        line = line.strip()
        data_items = line.split(" ")
        sent = data_items[1]
        start = int(data_items[2])
        length = int(data_items[3])
        tag = data_items[4]

        if(tag=='idiom'):
            f = open(os.path.join(sentence_root,sent),"r")
            sentence = f.read().strip()
            idiomic_part = sentence[start:start+length]
            print(sentence)
            print(idiomic_part)
            print(start,length)
            res_writer.writerow([sent, sentence, idiomic_part, start, length])

res_file.close()
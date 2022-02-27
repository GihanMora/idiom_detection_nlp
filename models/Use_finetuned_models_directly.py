from datasets import load_dataset
emotions = load_dataset("emotion")
#https://huggingface.co/datasets/emotion
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, AutoModel

model_name = r"E:\Projects\emo_detector_new\go_model"
tokenizer = AutoTokenizer.from_pretrained(r"E:\Projects\emo_detector_new\go_model")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

import ast
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support, \
    confusion_matrix
from transformers import AutoModelForSequenceClassification
num_labels = 28
model = (AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device))

print(emotions_encoded["train"].features)

emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
print(emotions_encoded["train"].features)


from sklearn.metrics import accuracy_score, f1_score

def compute_metrics_all(pred, ground_labels):
    labels_all = ground_labels
    preds_all = list(pred)

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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


from transformers import Trainer, TrainingArguments

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
training_args = TrainingArguments(output_dir=r"E:\Projects\emo_detector_new\results",
                                  num_train_epochs=8,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="f1",
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,)


from transformers import Trainer

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"])
# trainer.train();
import pandas as pd
# results = trainer.evaluate()
# print(results)
path = r"E:\Projects\emo_detector_new\datasets/ISEAR_dataset_cleaned.csv"
results_df = pd.DataFrame()
dff = pd.read_csv(path)
print(dff.columns)
print(dff['sentiment'].unique())
print(len(dff))
preds = []
sentences_zz = []
ground_truths = []
# p=0
for i,row in dff.iterrows():
    row_dict = row.to_dict()
    # print()
    sentence = row['text']
    try:
        ground_truths.append(int(row['sentiment_id']))
        # print(sentence)
        sentences_zz.append(sentence)
    except:
        continue

texts = sentences_zz
tokenized_texts = tokenizer(texts, padding=True)


class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}


test_dataset = SimpleDataset(tokenized_texts)



import numpy as np
# preds_output = trainer.predict(emotions_encoded["validation"])
preds_output = trainer.predict(test_dataset)
print(np.shape(preds_output))
print(preds_output.metrics)

import numpy as np
from sklearn.metrics import plot_confusion_matrix
y_valid = np.array(emotions_encoded["validation"]["label"])
y_preds = np.argmax(preds_output.predictions, axis=1)
print(y_preds)
# for x in y_preds:
#     print(x)

compute_metrics_all(pred=y_preds,ground_labels=ground_truths)
# labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
# plot_confusion_matrix(y_preds, y_valid, labels)
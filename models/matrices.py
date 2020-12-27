from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix


def compute_metrics(pred):

    labels = [1,1,1,0,0,0,0,-11,1,1,0,0]
    preds =  [1,1,1,0,0,0,101,0,0,0,0,0]
    # print(labels[1])
    # print(preds[1])
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    confusion_mat = confusion_matrix(labels, preds)


    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusiton_mat':confusion_mat
    }


output = compute_metrics([])
for k in output.keys():
    print(k)
    print(output[k])
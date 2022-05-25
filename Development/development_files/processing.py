import numpy as np
from transformer import Transformer

def int2oneHot(x_labels):

    x_labels = np.array(x_labels)
    y = []
    for label in x_labels:
        if label:
            y.append(np.array([0,1]))
        else:
            y.append(np.array([1,0]))
    return np.array(y)

class Preprocessing(Transformer):
    def __init__(self):
        super().__init__()

    def _prepareData(self, corpus):
        x_int = []
        x_mask = []
        x_labels = []

        for i in (range(len(corpus))):
            data = corpus[i][1]
            sentence = corpus[i][0]
            
            x_labels.append(data)
            ids,masks,segments = self._tokenize(sentence, pad_to_max_length=True)
            x_int.append(ids)
            x_mask.append(masks)
        
        y_labels = int2oneHot(x_labels)
        
        return np.array(x_int), np.array(x_mask), y_labels, np.array(x_labels)
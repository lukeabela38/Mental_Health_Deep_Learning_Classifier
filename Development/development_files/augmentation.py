from eda import eda

def dataAugmentation(labelled_corpus):

    sentences = []
    for entry in labelled_corpus:
        sentence = entry[0]
        label = entry[1]
        augmented_sentences = eda(sentence)
        for augmented_sentence in augmented_sentences:
            sentences.append([augmented_sentence, label])
    return sentences
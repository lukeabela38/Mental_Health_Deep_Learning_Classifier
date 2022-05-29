## TODO: imports
import argparse
import time
from data import Data
from augmentation import dataAugmentation
from processing import Preprocessing
from transformer import Transformer

def main():

    ## TODO: Load Data and return partitioned data set
    data = Data(verbose=False)
    data._prepareDict()
    labelled_data, label = data._prepareCorpus(N = int(input("Input Integer: ")))

    print("")
    print("Beginning Training of classifier for: ", label)
    print("")
    time.sleep(3)

    training_data, validation_data, test_data = data._dataSplit()


    print("Training Data: ", len(training_data), training_data[0])
    print("")
    print("Validation Data: ", len(validation_data), validation_data[0])
    print("")
    print("Testing Data: ", len(test_data), test_data[0])
    print("")

    ## TODO: Augment Training Dataset - Temporarily disabled due to latency
    # training_data = dataAugmentation(training_data)

    ## TODO: Processing of Datasets
    processor = Preprocessing()
    x_train_int, x_train_mask, y_train_labels, x_train_labels = processor._prepareData(training_data[:50])
    x_val_int, x_val_mask, y_val_labels, x_val_labels = processor._prepareData(validation_data[:50])
    x_test_int, x_test_mask, y_test_labels, x_test_labels = processor._prepareData(training_data[:50])

    print("X_TRAIN_INT: ", x_train_int[0])
    print("X_TRAIN_MASK: ", x_train_mask[0])
    print("Y_TRAIN_LABELS: ", len(y_train_labels))

    ## TODO: create Model based off Roberta

    transformer = Transformer()
    model = transformer._createModel_OOB()

    print(model.summary())
    print("")

    train_data = [x_train_int,x_train_mask,y_train_labels]
    validation_data = [x_val_int,x_val_mask, y_val_labels]
    test_data = [x_test_int,x_test_mask, y_test_labels]

    ## TODO: train and evaluate and saveModel

    history = transformer._trainModel(train_data, validation_data, epochs = 5, batch_size = 256)
    print("History: ", history)

    results = transformer._testModel(test_data)
    print("Results: ", results)

    transformer._saveModel(filepath=label)

if __name__ == "__main__":
    main()
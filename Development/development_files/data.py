import os
import glob
import csv
import random

class Data:
    def __init__(self, folder = "Datasets", verbose = True):
        self.location = os.getcwd()
        self.folder = "Datasets"
        self.verbose = verbose

    def _getFilePaths(self):
        return glob.glob(self.location + "/" + self.folder + "/*.csv") 

    def _loadSet(self, filename):

        rows = []   
        with open(filename, 'r') as csvfile: 
            reader = csv.reader(csvfile, delimiter=',') 
            for row in reader: 
                rows.append([row[0],row[3]])
        return rows[1:len(rows)]
    
    def _prepareDict(self):

        mental_health_data = []
        mental_health_data_names = []
        
        filepaths = self._getFilePaths()
        print("Filepaths: ", filepaths)
        for filepath in filepaths:

            corpus = self._loadSet(filepath)
            name = filepath[20:-27] # check filepath

            if self.verbose:
                print("")
                print("Dataset: ", name)
                print("Length of Dataset: ", len(corpus))
                print("")

            mental_health_data.append(corpus)
            mental_health_data_names.append(name)

        self.labels = mental_health_data_names
        self.dict = dict(zip(mental_health_data_names, mental_health_data))
        return self.dict

    def _prepareCorpus(self, N = 0):

        label = self.labels[N]

        corpus = self.dict[label]
        labelled_data = []

        positive_observations = []
        for i in range(len(corpus)):
            positive_observations.append([corpus[i][1], 1])
        

        negative_observations = []
        for label in self.labels:
            if label != self.labels[N]:

                corpus = self.dict[label]
                for i in range(len(corpus)):
                    negative_observations.append([corpus[i][1], 0])

        if self.verbose:
            print("")
            print("Length of Positive Observations before Data Balancing:  ", len(positive_observations))
            print("Length of Negative Observations before Data Balancing:  ", len(negative_observations))
            print("")
        ## Given the nature of this task negative observations are always more frequent than positive observations - hence undersample negative

        while (len(negative_observations) != len(positive_observations)):
            n = random.randint(0, len(negative_observations) - 1)
            negative_observations.pop(n)

        if self.verbose:
            print("")
            print("Length of Positive Observations after Data Balancing:  ", len(positive_observations))
            print("Length of Negative Observations after Data Balancing:  ", len(negative_observations))
            print("")

        assert len(positive_observations) == len(negative_observations)

        labelled_data = [*positive_observations, *negative_observations]

        random.shuffle(labelled_data)

        if self.verbose:
            print("")
            print("Data Checks: ")
            print(labelled_data[0])
            print(labelled_data[10])
            print(labelled_data[100])
            print(labelled_data[1000])
            print("")

        self.labelled_data = labelled_data

        return self.labelled_data, self.labels[N]

    def _dataSplit(self, ratio = [0.8, 0.1, 0.1]):

        N = len(self.labelled_data)

        self.training_data = self.labelled_data[:int(ratio[0]*N)]
        self.validation_data = self.labelled_data[int(ratio[0]*N): int((ratio[0]+ratio[1])*N)]
        self.test_data = self.labelled_data[int((ratio[0]+ratio[1])*N):]

        if self.verbose:
            print("Training Data: ", len(self.training_data), self.training_data[0])
            print("")
            print("Validation Data: ", len(self.validation_data), self.validation_data[0])
            print("")
            print("Testing Data: ", len(self.test_data), self.test_data[0])
            print("")

        return self.training_data, self.validation_data, self.test_data

if __name__ == "__main__":
    data = Data()
    data._prepareDict()

    labelled_data = data._prepareCorpus(N = 0)
    print("")
    print(labelled_data[0])
    print("Length: ", len(labelled_data))
    data._dataSplit()
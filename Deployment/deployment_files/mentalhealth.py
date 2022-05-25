
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import preprocess_corpus



class MentalHealth:
    def __init__(self):  
        with open('Models/Anxiety_Dict.p', 'rb') as handle:
            self.anxiety_word_index = pickle.load(handle)

        with open('Models/Control_Dict.p', 'rb') as handle:
            self.control_word_index = pickle.load(handle)

        with open('Models/Depression_Dict.p', 'rb') as handle:
            self.depression_word_index = pickle.load(handle)

        with open('Models/Autism_Dict.p', 'rb') as handle:
            self.autism_word_index = pickle.load(handle)

        with open('Models/BPD_Dict.p', 'rb') as handle:
            self.BPD_word_index = pickle.load(handle)

        with open('Models/Bipolar_Dict.p', 'rb') as handle:
            self.bipolar_word_index = pickle.load(handle)

        with open('Models/Schizo_Dict.p', 'rb') as handle:
            self.schizo_word_index = pickle.load(handle)


        self.anxiety_model = tf.keras.models.load_model('Models/MHSMC_Model_Anxiety')
        self.control_model = tf.keras.models.load_model('Models/MHSMC_Model_Control')
        self.depression_model = tf.keras.models.load_model('Models/MHSMC_Model_Depression')
        self.autism_model = tf.keras.models.load_model('Models/MHSMC_Model_Autism')
        self.BPD_model = tf.keras.models.load_model('Models/MHSMC_Model_BPD')
        self.bipolar_model = tf.keras.models.load_model('Models/MHSMC_Model_Bipolar')
        self.schizo_model = tf.keras.models.load_model('Models/MHSMC_Model_Schizo')


    def _runMentalHealthAlarmSystem(self, input_sentence):
        processed_sentence = preprocess_corpus(input_sentence.split(' '))

        x_anxiety = []
        x_control = []
        x_depression = []
        x_autism = []
        x_BPD = []
        x_bipolar = []
        x_schizo = []

        x_anxiety_numeric = []
        x_control_numeric = []
        x_depression_numeric = []
        x_autism_numeric = []
        x_BPD_numeric = []
        x_bipolar_numeric = []
        x_schizo_numeric = []

        for token in processed_sentence:
            x_anxiety_numeric.append(self.anxiety_word_index[token])
            x_control_numeric.append(self.control_word_index[token])
            x_depression_numeric.append(self.depression_word_index[token])
            x_autism_numeric.append(self.autism_word_index[token])
            x_BPD_numeric.append(self.BPD_word_index[token])
            x_bipolar_numeric.append(self.bipolar_word_index[token])
            x_schizo_numeric.append(self.schizo_word_index[token])

        x_anxiety.append(x_anxiety_numeric)
        x_control.append(x_control_numeric)
        x_depression.append(x_depression_numeric)
        x_autism.append(x_autism_numeric)
        x_BPD.append(x_BPD_numeric)
        x_bipolar.append(x_bipolar_numeric)
        x_schizo.append(x_schizo_numeric)

        max_sequence_length = 5000

        x_anxiety = np.array(pad_sequences(x_anxiety, maxlen = 4792, padding = 'pre'))
        x_control = np.array(pad_sequences(x_control, maxlen = 4792, padding = 'pre'))
        x_depression = np.array(pad_sequences(x_depression, maxlen = 8403, padding = 'pre'))
        x_autism = np.array(pad_sequences(x_autism, maxlen = 3902, padding = 'pre'))
        x_BPD = np.array(pad_sequences(x_BPD, maxlen = 2996, padding = 'pre'))
        x_bipolar = np.array(pad_sequences(x_bipolar, maxlen = 4792, padding = 'pre'))
        x_schizo = np.array(pad_sequences(x_schizo, maxlen = 2781, padding = 'pre'))

        mental_health_stats = []

        anxiety_percentage = (self.anxiety_model.predict(x_anxiety))
        control_percentage = (self.control_model.predict(x_control))
        depression_percentage = (self.depression_model.predict(x_depression))
        autism_percentage = (self.autism_model.predict(x_autism))
        BPD_percentage = (self.BPD_model.predict(x_BPD))
        bipolar_percentage = (self.bipolar_model.predict(x_bipolar))
        schizo_percentage = (self.schizo_model.predict(x_schizo))

        mental_health_stats.append("{:.2%}".format(anxiety_percentage[0][1]))
        mental_health_stats.append("{:.2%}".format(control_percentage[0][1]))
        mental_health_stats.append("{:.2%}".format(depression_percentage[0][1]))
        mental_health_stats.append("{:.2%}".format(autism_percentage[0][1]))
        mental_health_stats.append("{:.2%}".format(BPD_percentage[0][1]))
        mental_health_stats.append("{:.2%}".format(bipolar_percentage[0][1]))
        mental_health_stats.append("{:.2%}".format(schizo_percentage[0][1]))

        df = pd.DataFrame(mental_health_stats, columns = ['Percentage Chance of Disorder'])
        df['Disorder'] = ['Anxiety', 'Control', 'Depression', 'Autism', 'Borderline Personality', 'Bipolar', 'Schizo']
        df['Description'] = ['Assess similarity to Reddit discussion of Anxiety', 
                        'Assess similarity to Reddit forum to discuss, vent, support and share information about mental health, illness, and wellness.', 
                        'Assess similarity to Reddit discussion of Depression',
                        'Assess similarity to Reddit discussion of Autism',
                        'Assess similarity to Reddit discussion of Borderline Personality Disorder',
                        'Assess similarity to Reddit discussion of Bipolar',
                        'Assess similarity to Reddit disccusion of Schizo']
    
        return df

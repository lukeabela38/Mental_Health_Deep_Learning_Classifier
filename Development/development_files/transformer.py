import os
import numpy as np
import tensorflow as tf
from transformers import RobertaTokenizer 
from transformers import RobertaConfig, TFRobertaForSequenceClassification
from transformers import TFRobertaForSequenceClassification, TFRobertaModel
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D

class GlobalAveragePooling1DMasked(GlobalAveragePooling1D):
    def call(self, x, mask=None):
        if mask != None:
            return K.sum(x, axis=1) / K.sum(mask, axis=1)
        else:
            return super().call(x)


class Transformer:
    def __init__(self, max_length = 128, hdepth = 16, MAX_SEQUENCE_LENGTH=128, EMBED_SIZE=100, folder = "Models/"):

        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.ro_bert_a = 'roberta-base'
        self.max_length = max_length

        self.hdepth=hdepth
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.EMBED_SIZE=EMBED_SIZE

        # Defining DistilBERT tokonizer
        self.tokenizer = RobertaTokenizer.from_pretrained(self.ro_bert_a, do_lower_case=True, add_special_tokens=True,
                                                max_length=max_length, pad_to_max_length=True)

    def _tokenize(self, sentences, pad_to_max_length=True):
        if type(sentences) == str:
            inputs = self.tokenizer.encode_plus(sentences, add_special_tokens=True, max_length=self.max_length, pad_to_max_length=pad_to_max_length, 
                                                return_attention_mask=True, return_token_type_ids=True)
            return np.asarray(inputs['input_ids'], dtype='int32'), np.asarray(inputs['attention_mask'], dtype='int32'), np.asarray(inputs['token_type_ids'], dtype='int32')
            
        input_ids, input_masks, input_segments = [],[],[]
        for sentence in sentences:
            inputs = self.tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=self.max_length, pad_to_max_length=pad_to_max_length, 
                                                return_attention_mask=True, return_token_type_ids=True)
            input_ids.append(inputs['input_ids'])
            input_masks.append(inputs['attention_mask'])
            input_segments.append(inputs['token_type_ids'])        
            
        return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments, dtype='int32')

    def _create_RoBERTaForSequenceClassification(self):
        config = RobertaConfig(num_labels=2)
        config.output_hidden_states = False

        transformer_model = TFRobertaForSequenceClassification.from_pretrained(self.ro_bert_a)
        input_ids = tf.keras.layers.Input(shape=(self.max_length,), name='input_token', dtype='int32')
        input_masks_ids = tf.keras.layers.Input(shape=(self.max_length,), name='masked_token', dtype='int32')
        X = transformer_model(input_ids, input_masks_ids)

        return tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs = X)

    def _createModel_OOB(self):
        model = self._create_RoBERTaForSequenceClassification()
        self.model = model
        return self.model

    def _get_BERT_layer(self):
        ro_bert_a = 'roberta-base'
        config = RobertaConfig(dropout=0.2, attention_dropout=0.2)
        config.output_hidden_states = False
        return TFRobertaModel.from_pretrained(self.ro_bert_a)

    def _create_bag_of_words_BERT(self):

        input_ids_in = tf.keras.layers.Input(shape=(self.max_length,), name='input_token', dtype='int32')
        input_masks_in = tf.keras.layers.Input(shape=(self.max_length,), name='masked_token', dtype='int32') 

        bert_embeddings = self._get_BERT_layer()
        embedded_sent = bert_embeddings(input_ids_in, attention_mask=input_masks_in)[0]

        pooled_sent=GlobalAveragePooling1DMasked()(embedded_sent)
        hidden_output=Dense(self.hdepth,input_shape=(self.MAX_SEQUENCE_LENGTH,self.EMBED_SIZE),activation='sigmoid',kernel_initializer='glorot_uniform')(pooled_sent) # Sigmoid
        label=Dense(1,input_shape=(self.hdepth,),activation='sigmoid',kernel_initializer='glorot_uniform')(hidden_output)
        return Model(inputs=[input_ids_in,input_masks_in], outputs=[label],name='Model2_BERT')

    def _createModel_DAN(self):
        model = self._create_bag_of_words_BERT()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.model = model
        return self.model

    def _compileModel(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def _trainModel(self, train_data, validation_data, epochs = 5, batch_size = 1):

        self._compileModel()
        x_train_int,x_train_mask,x_train_labels = train_data
        x_val_int,x_val_mask,x_val_labels = validation_data

        history = self.model.fit((x_train_int,x_train_mask), x_train_labels, epochs=epochs, batch_size=batch_size, validation_data=([x_val_int,x_val_mask], x_val_labels), verbose=1)
        return history

    def _testModel(self, test_data):

        x_test_int,x_test_mask,x_test_labels = test_data
        result =  self.model.evaluate((x_test_int,x_test_mask), x_test_labels)
        return result

    def _saveModel(self, filepath = "MentalHealth"):

        print("Filepath to save model: ", filepath)
        filepath = self.folder + "/" + filepath
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        save_model(self.model, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None)

if __name__ == "__main__":

    transform = Transformer()
    inputs = transform._tokenize("The capital of France is [MASK].")
    print('INPUTS: ', inputs,'\n')

    inputs = transform._tokenize("This is a pretrained model.")
    print('INPUTS: ', inputs,'\n')

    ids,masks,segments = transform._tokenize("The capital of France is [MASK].")
    print('IDS: ', ids)
    print('MASKS: ', masks,"\n")
    print('SEGMENTS: ', segments)

    ids,masks,segments = transform._tokenize("The capital of France is [MASK].", pad_to_max_length=False)
    print('IDS: ', ids)
    print('MASKS: ',masks)
    print('SEGMENTS: ', segments)
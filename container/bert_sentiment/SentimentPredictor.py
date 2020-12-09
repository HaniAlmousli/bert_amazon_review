
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text



class SentimentPredictor(tf.keras.Model):
    def __init__(self,drop_out_rate=0.1,n_classes=5,
                 tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1',
                 tfhub_handle_encoder    = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'):
        super(SentimentPredictor, self).__init__()
        self.n_classes=n_classes
        self.preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
        self.encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        self.dropout_layer = tf.keras.layers.Dropout(drop_out_rate)
        self.out_layer = tf.keras.layers.Dense(n_classes, activation=None, name='classifier')

    def call(self, inp):
        self.encoder_inputs = self.preprocessing_layer(inp)
        #This is simply taking the pooled_output. It is possible to easily make it more complex
        #and get all the sequences and run them through more comple analysis (e.g RNN)
        self.bert_out = self.encoder(self.encoder_inputs)['pooled_output']
        x =self.dropout_layer(self.bert_out)
        self.final_output = self.out_layer(x)
        return self.final_output


#imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization, Embedding, Dense, GlobalAveragePooling1D, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential


def build_sequential(vectorize_layer, vocab_size, embedding_dim, sequence_length, dens_nn=164, dropout_n=0.2, activ_fn='relu'):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorize_layer,
        Embedding(vocab_size, embedding_dim, name="embedding"),
        Reshape((embedding_dim * sequence_length, ), name='concat_words'),
        Dense(dens_nn, activation=activ_fn, name='hidden_layer'),
        Dropout(dropout_n),
        Dense(1, name = 'output_layer')
    ])
    return model


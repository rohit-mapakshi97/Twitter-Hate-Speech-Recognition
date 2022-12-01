#!/usr/bin/python3
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, LSTM, Bidirectional, GlobalMaxPool1D, BatchNormalization, Flatten


def CNN(embedding_matrix, input_length):
    model = Sequential()
    # Embedding Layer
    model.add(Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=input_length))

    # CNN Layer (1 dimensional).  Valid padding is used to reduce dimensinoality
    model.add(Conv1D(filters=32, kernel_size=2,
              padding='valid', activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=2,
              padding='valid', activation='relu'))

    # Pooling Layer: Max pooling is used to reduce dimensionality
    model.add(GlobalMaxPooling1D())
    # Dense Layer
    # Dropout layer is used to reduce overfitting
    # Dense Layer for classification: 3 nodes for (hate_speech, offensive_language, neither)
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))  # classification layer
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def BILSTM(embedding_matrix, input_length):
    model = Sequential()
    # Embedding Layer
    model.add(Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=input_length))
    # Bidirectional LSTM
    model.add(Bidirectional(LSTM(
        input_length,
        return_sequences=True,
        recurrent_dropout=0.2)))
    # Pooling Layer: Max pooling is used to reduce dimensionality
    model.add(GlobalMaxPool1D())  # globalmaxpooling_layer
    # Batch normalization accelerates the learning process by reducing the number of epocs
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(input_length, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(input_length, activation="relu"))
    model.add(Dropout(0.5))
    # Classification
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def MultiLayerPerceptron(embedding_matrix, input_length):
    model = Sequential()
    model.add(Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=input_length))
    model.add(Flatten())  
    model.add(Dense(512, activation='relu'))  # dense_layer
     # Dropout layer is used to reduce overfitting
    model.add(Dropout(0.2))  
     # Dense Layer for classification: 3 nodes for (hate_speech, offensive_language, neither)
    model.add(Dense(3, activation='softmax')) 
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


# References
# Embedding Layer: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# CNNs: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
# Dropout Regularization: https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/
# Bi-directional LSTM: https://analyticsindiamag.com/complete-guide-to-bidirectional-lstm-with-python-codes/
# Batch Normalization: https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/

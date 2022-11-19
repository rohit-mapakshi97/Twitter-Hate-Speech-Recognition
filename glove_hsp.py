#!/usr/bin/python3
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from models_configurations_glove import CNN
import os


def getDatasetStats():
    pass

# Class weights are used when we have an imbalanced dataset. The under represented features
# will have higher error rate
# ref: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data


def getClassWeights(series):
    hate, ofensive, neither = np.bincount(series)
    total = hate + ofensive + neither
    weight_class1 = (1 / hate)*(total)/3.0
    weight_class2 = (1 / ofensive)*(total)/3.0
    weight_class3 = (1 / neither)*(total)/3.0
    return {0: weight_class1, 1: weight_class2, 2: weight_class3}


def preprocessText(line):
    # 1 Remove Retweet tag
    REGEX_RE_TWEET = '\sRT\s'
    line = re.sub(REGEX_RE_TWEET, ' ', line)  # note: replace with space

    # 2 Remove urls
    REGEX_URL = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|''[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    line = re.sub(REGEX_URL, '', line)

    # 3 Remove Twitter mentions
    REGEX_MENTION = '@[\w\-]+'
    line = re.sub(REGEX_MENTION, '', line)

    # 4 Remove special charecters
    REGEX_SPECIAL_CHARS = '[^A-Za-z0-9]+'
    line = re.sub(REGEX_SPECIAL_CHARS, ' ', line).strip()

    # 5 Change multiple spaces to one space
    REGEX_SPACE = '\s+'
    line = re.sub(REGEX_SPACE, ' ', line)

    # 6 Remove Stop Words
    tokens = [w for w in (nltk.word_tokenize(line)) if w.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    line = " ".join(tokens).lower()
    return line

# Glove file is formatted as: Word V1 V2 . . Vn


def getGloveEmbeddings():
    glove_dictionary = {}
    with open('res/glove.twitter.27B.100d.txt') as file:
        for line in file.readlines():
            records = line.split()
            word = records[0]
            vectors = np.asarray(records[1:], dtype='float32')
            glove_dictionary[word] = vectors
    return glove_dictionary, 100  # this glove file has 100D


def getTextSequences(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    longest_tweet = max(
        texts, key=lambda sentence: len(word_tokenize(sentence)))
    length_long_tweet = len(word_tokenize(longest_tweet))

    def genSequence(text_data): return tokenizer.texts_to_sequences(text_data)

    # Text Sequences with padding
    train_padded_sentences = pad_sequences(
        genSequence(texts),
        length_long_tweet,
        padding='post'
    )
    return train_padded_sentences, tokenizer


TEXT = "tweet"
LABEL = "class"
CLASS_HATE_SPEECH = "Hate Speech"
CLASS_OFFENSIVE_LANGUAGE = "Offensive Language"
CLASS_NEITHER = "Neither"

class_map = {0: CLASS_HATE_SPEECH,
             1: CLASS_OFFENSIVE_LANGUAGE, 2: CLASS_NEITHER}

PATH_CNN_GLOVE = "models/glove_cnn"
PATH_TWEETS_DATA = "data/t_davison_twitter_labeled_data.csv" #"data/test.csv"

if __name__ == "__main__":
    raw_df = pd.read_csv(PATH_TWEETS_DATA, decimal=",")
    df = pd.concat([raw_df[TEXT], raw_df[LABEL]], axis=1)
    # Step 1 Preprocess:
    print("\n1. Preprocessing Tweets..\n")
    df[TEXT] = df.apply(lambda x: preprocessText(x[TEXT]), axis=1)

    # Step 2 Generate Word Embeddings
    print("2. Generating Glove Embeddings..\n")
    texts = df[TEXT]
    glove_dictionary, glove_dim = getGloveEmbeddings()

    # 2.1 Generate word sequences
    train_padded_sentences, tokenizer = getTextSequences(texts)
    vocab_length = len(tokenizer.word_index) + 1

    # words not in the glove dictionary will have vector = 0
    embedding_matrix = np.zeros((vocab_length, glove_dim))
    for word, index in tokenizer.word_index.items():
        embedding_vector = glove_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    # Step 3 Train
    # 3.1 Generate Train Test Validation Split
    print("3. Training Models")
    labels = df[LABEL]
    X_train, X_test, Y_train, Y_test = train_test_split(
        train_padded_sentences,
        labels,
        test_size=0.25
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train,
        Y_train,
        test_size=0.1)

    class_weights = getClassWeights(labels)
    input_length = train_padded_sentences.shape[1]
    # epoch_count = 100
    # batch_size = 128

    # Reduce learning rate when model reaches a plateau
    # ref: https://keras.io/api/callbacks/reduce_lr_on_plateau/
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, verbose=1, patience=5, min_lr=0.001)

    print("3.1 Training CNN\n")
    cnn_classifer = None
    if (os.path.isdir(PATH_CNN_GLOVE)):
        cnn_classifer = load_model(PATH_CNN_GLOVE)
    else:
        cnn_classifer = CNN(embedding_matrix, input_length)
        history_cnn = cnn_classifer.fit(
            X_train,
            Y_train,
            epochs=75,
            batch_size=128,
            validation_data=(X_val, Y_val),
            verbose=1,
            callbacks=[reduce_lr],
            class_weight=class_weights
        )
        cnn_classifer.save(PATH_CNN_GLOVE)
    print(cnn_classifer.summary())

    # Step 4 Test
    cnn_predict = np.argmax(cnn_classifer.predict(X_test), axis=-1)
    print(classification_report(Y_test, cnn_predict))
    print(confusion_matrix(Y_test, cnn_predict))

    # Step 5 Generate Stats
    pass

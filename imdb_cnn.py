import sys
import imdb

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, Flatten, MaxPooling1D
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import wandb
from wandb.keras import WandbCallback
from keras.preprocessing import text
import numpy as np

def main():
    # Download and Extract IMDB dataset archive
    imdb.download_and_extract_imdb()

    #
    wandb.init()
    config = wandb.config

    # set parameters:
    config.vocab_size = 1000
    config.maxlen = 1000
    config.batch_size = 30
    config.embedding_dims = 10
    config.filters = 16
    config.kernel_size = 3
    config.hidden_dims = 256
    config.epochs = 10

    (X_train, y_train), (X_test, y_test) = imdb.load_imdb()

    print("===Train===")
    print(f"X_train shape: {len(X_train)}")
    print(f"y_train shape: {len(y_train)}")

    print("===Test===")
    print(f"X_test shape: {len(X_test)}")
    print(f"y_test shape: {len(y_test)}")

    tokenizer = text.Tokenizer(num_words=config.vocab_size)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_matrix(X_train)
    X_test = tokenizer.texts_to_matrix(X_test)

    X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen)

    model = Sequential()
    model.add(Embedding(config.vocab_size,
                        config.embedding_dims,
                        input_length=config.maxlen))
    model.add(Dropout(0.5))
    model.add(Conv1D(config.filters,
                    config.kernel_size,
                    padding='valid',
                    activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(config.filters,
                    config.kernel_size,
                    padding='valid',
                    activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(config.hidden_dims, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    print("===Summary===")
    model.summary()

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(X_train, y_train,
            batch_size=config.batch_size,
            epochs=config.epochs,
            validation_data=(X_test, y_test), callbacks=[WandbCallback()])

    
    # predict probabilities for test set
    yhat_probs = model.predict(X_test, verbose=0)
    yhat_probs = yhat_probs[:, 0]
    # predict crisp classes for test set
    yhat_classes = np.rint(yhat_probs)

    print(f"yhat_probs: {yhat_probs}")

    print(f"yhat_classes: {yhat_classes}")
    
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    print('F1 score: %f' % f1)

if __name__ == "__main__":
    sys.exit(main())

import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

PATH_TO_FILE = "datasets/news.csv"
MAX_VOCAB = 10000

def get_bi_gru_improved_model(embedding_layer_length):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(embedding_layer_length, 256),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True), name="Bidirectional_LSTM_1"),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16),  name="Bidirectional_LSTM_2"),
        tf.keras.layers.Dense(16, name="Hidden"),
        tf.keras.layers.Activation("relu", name="Activation"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, name="Output")
    ])

def get_bi_gru_model(embedding_layer_length):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(embedding_layer_length, 256, name="Embedding"),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32), name="Bidirectional_GRU"),
        tf.keras.layers.Dense(256, name="Hidden"),
        tf.keras.layers.Activation("relu", name="Activation"),
        tf.keras.layers.Dense(1, name="Output"),
    ])

def get_bi_lstm_model(embedding_layer_length):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(embedding_layer_length, 256, name="Embedding"),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32), name="Bidirectional_LSTM"),
        tf.keras.layers.Dense(256, name="Hidden"),
        tf.keras.layers.Activation("relu", name="Activation"),
        tf.keras.layers.Dense(1, name="Output"),
    ])

def get_training_and_testing_sequences(train, test, tokenizer):
    X_train = tokenizer.texts_to_sequences(train)
    X_test = tokenizer.texts_to_sequences(test)

    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=256)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', maxlen=256)

    return X_train, X_test

if __name__ == "__main__":
    df: pd.DataFrame = pd.read_csv(PATH_TO_FILE)

    X_train, X_test, y_train, y_test = train_test_split(df['title'], df['truth'], test_size=0.2, random_state=5122022)
    tokenizer = Tokenizer(num_words=MAX_VOCAB)
    tokenizer.fit_on_texts(X_train)
    X_train, X_test = get_training_and_testing_sequences(X_train, X_test, tokenizer)

    model = get_bi_gru_improved_model(MAX_VOCAB)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=['accuracy']
    )

    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=30, shuffle=True, callbacks=[early_stop])

    pred_norm = []

    for prob in model.predict(X_test):
        pred_norm.append(int(prob >= 0.5))

    print('Accuracy on testing set:', accuracy_score(pred_norm, y_test))
    print('Precision on testing set:', precision_score(pred_norm, y_test))

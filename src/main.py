
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import os
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

PATH_TO_FILE = "datasets/news_norm.csv"
OUT_DIR = "out"
TEXT_ATTRIBUTE = "norm_title"

TRANSLATE = {
    "multinomial_nb": "Multinomial Naive Bayes",
    "bernoulli_nb": "Bernoulli Naive Bayes",
}

def generate_wordcloud(words, file_name = ''):
    wordcloud = WordCloud(
                    width = 800,
                    height = 400,
                    background_color ='white',
                    min_font_size = 12,
                ).generate(words)
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(file_name)


if __name__ == "__main__":
    df: pd.DataFrame = pd.read_csv(PATH_TO_FILE)

    for name, group in df.groupby('truth'):
        group[TEXT_ATTRIBUTE] = group[TEXT_ATTRIBUTE].apply(literal_eval).apply(lambda x: ' '.join(x))
        all_words = ' '.join(group[TEXT_ATTRIBUTE])
        generate_wordcloud(all_words, f'{OUT_DIR}/word_cloud_{name}')

    df[TEXT_ATTRIBUTE] = df[TEXT_ATTRIBUTE].apply(literal_eval).apply(lambda x: ' '.join(x))
    all_words = ' '.join(df[TEXT_ATTRIBUTE])

    random_states = [42, 59, 81, 90, 156, 200, 215, 456, 9875, 1354]

    classes = {
        "multinomial_nb": MultinomialNB,
        "bernoulli_nb": BernoulliNB
    }

    for key in classes.keys():
        accuracy_array = []
        precision_array = []
        
        for value in random_states:
            X_train, X_test, y_train, y_test = train_test_split(df[TEXT_ATTRIBUTE], df['truth'], test_size=0.1, random_state=value)

            model = make_pipeline(TfidfVectorizer(), classes[key]())
            model.fit(X_train, y_train)
            predicted_categories = model.predict(X_test)

            accuracy_array.append(accuracy_score(y_test, predicted_categories))
            precision_array.append(precision_score(y_test, predicted_categories))
        
        cm = confusion_matrix(y_test, predicted_categories, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot()

        plt.savefig(f'{OUT_DIR}/confussion_matrix_{key}')
        
        fig, axis = plt.subplots()
        axis.set_title(f'Accuracy and Precision with {TRANSLATE[key]}')
        axis.boxplot([accuracy_array, precision_array],  labels=("Accuracy","Precision"))
        plt.savefig(f'{OUT_DIR}/boxplot_{key}')

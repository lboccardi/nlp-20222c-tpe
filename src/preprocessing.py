import nltk
nltk.download(['stopwords', 'punkt'])
import pandas as pd

PATH_TO_FILE = "datasets/news.csv"
stop_words = set(nltk.corpus.stopwords.words())

def normalize(string_1, string_2 = ''):
    title_tokenized_words = nltk.word_tokenize(string_1 + string_2)
    return [w.lower() for w in title_tokenized_words if not (w.lower() in stop_words) and (w.lower().isalnum())]

if __name__ == "__main__":

    ## This lines are just for generating the news.csv from Real and Fake datasets
    
    # df_fake: pd.DataFrame = pd.read_csv('datasets/Fake.csv', header=0)
    # df_true: pd.DataFrame = pd.read_csv('datasets/True.csv', header=0)

    # df_fake = df_fake.assign(truth=0)
    # df_true = df_true.assign(truth=1)

    # df_news = pd.concat([df_fake, df_true], ignore_index=True)

    # df_news.to_csv('./datasets/news.csv', index=False)

    df: pd.DataFrame = pd.read_csv(PATH_TO_FILE)
    df['norm_title'] = df.apply(lambda row: normalize(row['title']), axis=1)
    df['norm_text'] = df.apply(lambda row: normalize(row['text']), axis=1)
    df[['norm_text', 'norm_title', 'truth']].to_csv("datasets/news_norm.csv", index=False)

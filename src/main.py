
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval
import os
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

PATH_TO_FILE = "datasets/news_norm.csv"

def generate_wordcloud(words, file_name = ''):
    
    wordcloud = WordCloud(
                    width = 800,
                    height = 400,
                    background_color ='white',
                    min_font_size = 12,
                ).generate(words)
    os.makedirs('out', exist_ok=True)
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(file_name)


if __name__ == "__main__":
    df: pd.DataFrame = pd.read_csv(PATH_TO_FILE)

    for name, group in df.groupby('truth'):
        group['norm_title'] = group['norm_title'].apply(literal_eval).apply(lambda x: ' '.join(x))
        all_words = ' '.join(group['norm_title'])
        generate_wordcloud(all_words, f'out/word_cloud_{name}')
        


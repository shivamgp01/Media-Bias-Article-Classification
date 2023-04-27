import numpy as np
import pandas as pd
import os
import glob
import newspaper
from newspaper import Article
import seaborn
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def cleanup():
    sw_nltk = stopwords.words('english')
    mbicPhrases = mbic_clean(sw_nltk)
    kagglePhrases = kaggle_mediabias_clean(sw_nltk)
    
    return kagglePhrases, mbicPhrases
   
def mbic_clean(sw_nltk):
    #reads and gets rid of empty rows
    labeled_news = pd.DataFrame()
    filename = "./Data/mbic/labeled_dataset.csv"
    labeled_news = pd.read_csv(filename)
    labeled_news = labeled_news[labeled_news['article'].notna()]
    
    overallBiasedWords = np.array([])
    cats = np.array([])
    temp = pd.DataFrame()
   
    #get biased words and categories for them
    for index, row in labeled_news.iterrows():
        catTemp = [row['topic']]
        words = row['biased_words4']
        classification = row['Label_bias']
        words = (words[1:-1]).split(", ")
        for i in np.arange(len(words)):
            words[i] = words[i][1:-1]
        if(words[0] != ''):
            cats = np.append(cats, catTemp*len(words))
            overallBiasedWords = np.append(overallBiasedWords, words)
         
    temp['PHRASE'] = overallBiasedWords
    temp['CATEGORY_NAME'] = cats
    temp.sort_values('PHRASE')
    temp = temp[temp.columns.drop(list(temp.filter(regex='Test')))]
    
    overallBiasedWords.sort()
    overallBiasedWords = np.unique(overallBiasedWords)
    
    return temp 

def kaggle_mediabias_clean(sw_nltk):
    blacklist = pd.read_csv('Data/mediabiasKaggle/phrasebias_data/blacklist.csv', header=None)
    blacklist.rename(columns={0: 'blacklist'}, inplace=True)
    directory = "./Data/mediabiasKaggle/phrasebias_data/phrase_selection/"
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    phrases = pd.DataFrame()
    phrases
    categoryNumber = 0
    for f in csv_files: 
        tempDF = pd.read_csv(f)
        tempDF["CATEGORY_NAME"] = ((f.split("/"))[-1])[0:-12]
        
        phrases = pd.concat([phrases, tempDF[['PHRASE', 'CATEGORY_NAME']]], ignore_index=True)
        categoryNumber += 1
    
    return phrases

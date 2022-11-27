#requirements
import numpy as np
import pandas as pd 

from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def download_nltk_requirements(requirements=['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4']):
    for requirement in requirements:
        nltk.download(requirement)


def process_sentences(text):
    temp_sent =[]

    # Tokenize words
    words = nltk.word_tokenize(text)

    # Lemmatize each of the words based on their position in the sentence
    tags = nltk.pos_tag(words)
    for i, word in enumerate(words):
        if tags[i][1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'):  # only verbs
            lemmatized = lemmatizer.lemmatize(word, 'v')
        else:
            lemmatized = lemmatizer.lemmatize(word)
        
        # Remove stop words and non alphabet tokens
        if lemmatized not in stop_words and lemmatized.isalpha(): 
            temp_sent.append(lemmatized)

    # Some other clean-up
    full_sentence = ' '.join(temp_sent)
    full_sentence = full_sentence.replace("n't", " not")
    full_sentence = full_sentence.replace("'m", " am")
    full_sentence = full_sentence.replace("'s", " is")
    full_sentence = full_sentence.replace("'re", " are")
    full_sentence = full_sentence.replace("'ll", " will")
    full_sentence = full_sentence.replace("'ve", " have")
    full_sentence = full_sentence.replace("'d", " would")
    
    return full_sentence


def create_tfidf_maxtirx(df, target_col, ngram_range=(1,2), min_df=0):
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, min_df=min_df, stop_words='english')
    tfidf_maxtix = tfidf.fit_transform(df[target_col])

    cos_sim = linear_kernel(tfidf_maxtix, tfidf_maxtix)

    return cos_sim

def recommend_by_name(df,name, target_col, N, columns,do_ascending=False, rating="rating", name_col="name"):
    recommend = []

    df_reindexed = df.set_index(name_col)
    
    indices = pd.Series(df_reindexed.index)
    idx = indices[indices == name].index[0]
    cos_sim = create_tfidf_maxtirx(df_reindexed, target_col)

    score_series = pd.Series(cos_sim[idx]).sort_values(ascending=do_ascending)

    # Extract top N restuarnt
    top_N_indexes = list(score_series.iloc[0:30].index)
    
    for each in top_N_indexes:
        recommend.append(list(df_reindexed.index)[each])
    
    df_new = pd.DataFrame(columns=columns)

    for each in recommend:
        df_new = df_new.append(pd.DataFrame(df_reindexed[columns][df_reindexed.index == each].sample()))

    df_new = df_new.drop_duplicates(subset=columns, keep=False)
    df_new = df_new.sort_values(by=rating, ascending=do_ascending).head(N)
    
    return df_new

def recommend_by_description(df, description ,N, target_columns, columns, do_ascending=False):
    description = description.lower()
    description = process_sentences(description)
    description = description.strip()

    df['bag_of_words'] = pd.Series("", index=df.index)

    for column in target_columns:
        df['bag_of_words'] += (df[column].apply(process_sentences) + " ")

    # Init a TF-IDF vectorizer
    tfidfvec = TfidfVectorizer()    

    vec = tfidfvec.fit(df['bag_of_words'])
    features = vec.transform(df['bag_of_words'])

    # Transform user input data based on fitted model
    description_vector =  vec.transform([description])

    # Calculate cosine similarities between users processed input and reviews
    cos_sim = linear_kernel(description_vector, features)

    df['similarity'] = cos_sim[0]

    df.sort_values(by='similarity', ascending=do_ascending, inplace=True)
    df_new = df[~df.index.duplicated(keep='last')]

    return df_new[columns]
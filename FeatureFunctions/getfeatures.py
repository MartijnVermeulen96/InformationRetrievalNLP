import pandas as pd
from FeatureFunctions import helper
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from afinn import Afinn
import numpy as np

def getaffinfeats(tweets):
    afinn = Afinn()
    overall_score = tweets.apply(afinn.score)
    trainingdata_bags = helper.allbags(tweets)
    training_wordlength = [len(item) for item in trainingdata_bags]

    seperate_scores = []
    for i in range(len(tweets)):
        seperate_scores.append([afinn.score(word) for word in trainingdata_bags[i]])

    positive_words = [sum([1 for i in row if i > 0]) for row in seperate_scores]
    negative_words = [sum([1 for i in row if i < 0]) for row in seperate_scores]
    neutral_words = [sum([1 for i in row if i == 0]) for row in seperate_scores]

    difference_maxmin = [max(row) - min(row) if len(row) > 0 else 0 for row in seperate_scores]
    polarity_contrast = [a*b>0 for a,b in zip(negative_words,positive_words)]
    afinn_features = np.column_stack((positive_words, negative_words, neutral_words, overall_score, difference_maxmin, polarity_contrast))
    return afinn_features

def getlexical(train_data, tweet_column, unicount_vect=None, bicount_vect=None, tricount_vect=None, fourcount_vect=None):
    train_tweets = train_data[tweet_column]
    ##### Generate training features
    ### Getting all word bags
    trainingdata = pd.DataFrame();

    ### Bag of words
    Bags = helper.allbags(train_tweets)

    ### Getting PunctuationFlood features
    trainingdata["PunctuationFlood"] = helper.puncflood(train_tweets)

    ### Getting character flooding features

    trainingdata["CharFlood"] = helper.charflood(Bags, trainingdata["PunctuationFlood"])

    ### Getting Capitalized count feature

    trainingdata["CapitalizedCount"] = helper.capitalizefeature(Bags)

    ### Getting Punctuation is last features

    trainingdata["FinalPunctuation"] = [x[-1:] in string.punctuation for x in train_data[tweet_column]]

    ### Getting Hashtag feature

    trainingdata["HashtagCount"] = helper.hashtag(Bags)


    ### Getting Tweet Length features

    trainingdata["TweetCharLength"] = train_data[tweet_column].apply(len)
    trainingdata["TweetWordLength"] = [len(item) for item in Bags]


    ### Getting Hashtag 2 Word Ratio feature

    trainingdata["Hashtag2WordRatio"] = helper.hashwordratio(trainingdata["TweetWordLength"], trainingdata["HashtagCount"])


    ### Getting Emoji features

    trainingdata["EmojiCount"] = helper.emojifeats(Bags)

    ### Create feature vectors for *gram features
    if unicount_vect==None:
        unicount_vect = CountVectorizer(strip_accents="unicode", stop_words="english",
            analyzer=lambda x:x
        )
        UnigramVector = unicount_vect.fit_transform(Bags)
    else:
        UnigramVector = unicount_vect.transform(Bags)

    ### Getting all bigrams
    Bigrams = helper.bigrams(Bags)
    if bicount_vect==None:
        bicount_vect = CountVectorizer(strip_accents="unicode", stop_words="english",
            analyzer=lambda x:x
        )
        BigramVector = bicount_vect.fit_transform(Bigrams)
    else:
        BigramVector = bicount_vect.transform(Bigrams)

    ### Getting Character tri- and fourgrams
    CharTrigram = helper.chartrigrams(Bags)
    if tricount_vect==None:
        tricount_vect = CountVectorizer(strip_accents="unicode", stop_words="english",
            analyzer=lambda x:x
        )
        CharTrigramVector = tricount_vect.fit_transform(CharTrigram)
    else:
        CharTrigramVector = tricount_vect.transform(CharTrigram)

    CharFourgram = helper.charfourgrams(Bags)
    if fourcount_vect==None:
        fourcount_vect = CountVectorizer(strip_accents="unicode", stop_words="english",
            analyzer=lambda x:x
        )
        CharFourgramVector = fourcount_vect.fit_transform(CharFourgram)
    else:
        CharFourgramVector = fourcount_vect.transform(CharFourgram)


    ### Voeg de vectors toe aan DataFrame

    trainingdata['BigramVector'] = BigramVector.toarray().tolist()
    trainingdata['UnigramVector'] = UnigramVector.toarray().tolist()
    trainingdata['CharTrigramVector'] = CharTrigramVector.toarray().tolist()
    trainingdata['CharFourgramVector'] = CharFourgramVector.toarray().tolist()
    return trainingdata, unicount_vect, bicount_vect, tricount_vect, fourcount_vect

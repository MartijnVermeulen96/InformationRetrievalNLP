import pandas as pd
from FeatureFunctions import helper
import string
from sklearn.feature_extraction.text import CountVectorizer


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
        unicount_vect = CountVectorizer(
            analyzer=lambda x:x
        )
        UnigramVector = unicount_vect.fit_transform(Bags)
    else:
        UnigramVector = unicount_vect.transform(Bags)

    ### Getting all bigrams
    Bigrams = helper.bigrams(Bags)
    if bicount_vect==None:
        bicount_vect = CountVectorizer(
            analyzer=lambda x:x
        )
        BigramVector = bicount_vect.fit_transform(Bigrams)
    else:
        BigramVector = bicount_vect.transform(Bigrams)

    ### Getting Character tri- and fourgrams
    CharTrigram = helper.chartrigrams(Bags)
    if tricount_vect==None:
        tricount_vect = CountVectorizer(
            analyzer=lambda x:x
        )
        CharTrigramVector = tricount_vect.fit_transform(CharTrigram)
    else:
        CharTrigramVector = tricount_vect.transform(CharTrigram)

    CharFourgram = helper.charfourgrams(Bags)
    if fourcount_vect==None:
        fourcount_vect = CountVectorizer(
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

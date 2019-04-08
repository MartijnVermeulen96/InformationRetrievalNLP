import pandas as pd
from FeatureFunctions import helper
import string
from sklearn.feature_extraction.text import CountVectorizer


def getlexical(train_data, tweet_column):
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

    count_vect = CountVectorizer(
        analyzer=lambda x:x
    )

    ### Getting all bigrams
    Bigrams = helper.bigrams(Bags)

    ### Getting Character tri- and fourgrams
    CharTrigram = helper.chartrigrams(Bags)
    CharFourgram = helper.charfourgrams(Bags)

    BigramVector = count_vect.fit_transform(Bigrams)
    UnigramVector = count_vect.fit_transform(Bags)
    CharTrigramVector = count_vect.fit_transform(CharTrigram)
    CharFourgramVector = count_vect.fit_transform(CharFourgram)

    ### Voeg de vectors toe aan DataFrame

    trainingdata['BigramVector'] = BigramVector.toarray().tolist()
    trainingdata['UnigramVector'] = UnigramVector.toarray().tolist()
    trainingdata['CharTrigramVector'] = CharTrigramVector.toarray().tolist()
    trainingdata['CharFourgramVector'] = CharFourgramVector.toarray().tolist()
    return trainingdata

from nltk import ngrams
from itertools import groupby
import string
from emoji import UNICODE_EMOJI
from nltk.tokenize import TweetTokenizer
import numpy as np
from gensim.models import Word2Vec
import twokenize

def getindices(trainingdata, train_split=9, test_split=1):
    ### Creating 50-50% balance
    ## Train

    ### Count the amount of samples
    amount_nonirony_train = sum(trainingdata["Label"] == 0)
    amount_irony_train = sum(trainingdata["Label"] > 0)
    amount_train_amount = min(amount_nonirony_train, amount_irony_train)



    ### Sample indices
    nonirony_index = trainingdata[trainingdata["Label"] == 0].index.to_series()
    irony_index = trainingdata[trainingdata["Label"] > 0].index.to_series()

    total_nonirony_samples = nonirony_index.sample(amount_train_amount).tolist()
    total_irony_samples = irony_index.sample(amount_train_amount).tolist()

    train_amount = round(amount_train_amount / (train_split + test_split) * train_split)
    test_amount = round(amount_train_amount / (train_split + test_split) * test_split)

    resulting_train_index = total_nonirony_samples[:train_amount] + total_irony_samples[:train_amount]
    resulting_test_index = total_nonirony_samples[train_amount+1:] + total_irony_samples[train_amount+1:]

    return resulting_train_index, resulting_test_index


### Bags as input
def onehotwordclusters(bags, kmeansmodel, w2vmodel):
    vocab = w2vmodel.wv.index2word

    resulting_clusters = []

    for row in bags:
        filtered_words = list(filter(lambda x: x in vocab, row))
        if len(filtered_words)>0:
            result = list(kmeansmodel.predict(w2vmodel.wv[filtered_words]))
        else:
            result = []
        resulting_clusters.append(result)

    onehotvec = np.zeros((len(resulting_clusters), 2000))
    for i in range(len(resulting_clusters)):
        onehotvec[i, resulting_clusters[i]] = 1

    return onehotvec



def allbags(tweets):
  allbags = []
  for i in range(len(tweets)):
      bagofwords = twokenize.simpleTokenize(tweets[i])
      allbags.append(bagofwords)
  return allbags

def bigrams(bags):
    allbigrams = []
    for i in range(len(bags)):
        bigram = ngrams(bags[i],2)
        allbigrams.append(bigram)
    bigramlist=[]
    for i in range(len(bags)):
        value = list(bags[i])
        bigramlist.append(value)
    return bigramlist

def chartrigrams(bags):
    n=3
    chartrigram =[]
    for i in range(len(bags)):
        holdgram = []
        for j in range(len(bags[i])):
            holdgram.extend([bags[i][j][k:k+n] for k in range(len(bags[i][j])-n+1)])
        chartrigram.append(holdgram)
    return chartrigram

def charfourgrams(bags):
    n=4
    charfourgram =[]
    for i in range(len(bags)):
        holdgram = []
        for j in range(len(bags[i])):
            holdgram.extend([bags[i][j][k:k+n] for k in range(len(bags[i][j])-n+1)])
        charfourgram.append(holdgram)
    return charfourgram

def puncflood(tweets):
    ## ToDo: http:// is not flooding
    puncfloodfeature =[]
    for i in range(len(tweets)):
        counter = 0
        s = tweets[i]
        t = [[k,len(list(g))] for k, g in groupby(s)]
        for element in t:
            char = element[0]
            count = element[1]
            if char in string.punctuation and count > 1:
                counter += 1
        puncfloodfeature.append(counter)
    return puncfloodfeature


def charflood(bags, puncfloods):
    charfloodfeature = []
    for i in range(len(bags)):
        charflood = 0
        for j in range(len(bags[i])):
            b = [[r, len(list(g))] for r, g in groupby(bags[i][j])]
            if (max(b)[1]>2):
                charflood += 1
        charfloodfeature.append(charflood)
    charfloodfeature = [max(0,x1 - x2) for (x1, x2) in zip(charfloodfeature, puncfloods)]
    return charfloodfeature

def capitalizefeature(bags):
    capitalizefeature=[]
    for i in range(len(bags)):
        capfeat = 0
        for j in range(len(bags[i])):
            if (bags[i][j][0].isupper()):
                capfeat += 1
        capitalizefeature.append(capfeat)
    return capitalizefeature

def hashtag(bags):
    hashtagfeature=[]
    for i in range(len(bags)):
        hashtags = 0
        for j in range(len(bags[i])):
            if (bags[i][j][0] == "#"):
                hashtags += 1
        hashtagfeature.append(hashtags)
    return hashtagfeature

def hashwordratio(wordlengths, hashtagfeature):
    hashtagtowordfeature=hashtagfeature/wordlengths
    return hashtagtowordfeature

### Getting Emoji features

def is_emoji(s):
    count = 0
    for emoji in UNICODE_EMOJI:
        count += s.count(emoji)
        if count > 1:
            return False
    return bool(count)

def emojifeats(bags):
    emojifeature=[]
    for i in range(len(bags)):
        emojicount = 0
        for j in range(len(bags[i])):
            if (is_emoji(bags[i][j])):
                emojicount += 1;
        emojifeature.append(emojicount)
    return emojifeature


def calc_accuracy(true, predicted):
    """Calculates the accuracy of a (multiclass) classifier, defined as the fraction of correct classifications."""
    return sum([t==p for t,p in zip(true, predicted)]) / float(len(true))

def precision_recall_fscore(true, predicted, beta=1, labels=None, pos_label=None, average=None):
    """Calculates the precision, recall and F-score of a classifier.
    :param true: iterable of the true class labels
    :param predicted: iterable of the predicted labels
    :param beta: the beta value for F-score calculation
    :param labels: iterable containing the possible class labels
    :param pos_label: the positive label (i.e. 1 label for binary classification)
    :param average: selects weighted, micro- or macro-averaged F-score
    """

    # Build contingency table as ldict
    ldict = {}
    for l in labels:
        ldict[l] = {"tp": 0., "fp": 0., "fn": 0., "support": 0.}

    for t, p in zip(true, predicted):
        if t == p:
            ldict[t]["tp"] += 1
        else:
            ldict[t]["fn"] += 1
            ldict[p]["fp"] += 1
        ldict[t]["support"] += 1

    # Calculate precision, recall and F-beta score per class
    beta2 = beta ** 2
    for l, d in ldict.items():
        try:
            ldict[l]["precision"] = d["tp"]/(d["tp"] + d["fp"])
        except ZeroDivisionError: ldict[l]["precision"] = 0.0
        try: ldict[l]["recall"]    = d["tp"]/(d["tp"] + d["fn"])
        except ZeroDivisionError: ldict[l]["recall"]    = 0.0
        try: ldict[l]["fscore"] = (1 + beta2) * (ldict[l]["precision"] * ldict[l]["recall"]) / (beta2 * ldict[l]["precision"] + ldict[l]["recall"])
        except ZeroDivisionError: ldict[l]["fscore"] = 0.0

    # If there is only 1 label of interest, return the scores. No averaging needs to be done.
    if pos_label:
        d = ldict[pos_label]
        return (d["precision"], d["recall"], d["fscore"])
    # If there are multiple labels of interest, macro-average scores.
    else:
        for label in ldict.keys():
            avg_precision = sum(l["precision"] for l in ldict.values()) / len(ldict)
            avg_recall = sum(l["recall"] for l in ldict.values()) / len(ldict)
            avg_fscore = sum(l["fscore"] for l in ldict.values()) / len(ldict)
        return (avg_precision, avg_recall, avg_fscore)

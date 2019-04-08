import twokenize
from nltk import ngrams
from itertools import groupby
import string
from emoji import UNICODE_EMOJI

def allbags(tweets):
  allbags = []
  for i in range(len(tweets)):
    bagofwords = twokenize.simpleTokenize(tweets.iloc[i].lower())
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

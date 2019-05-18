# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Created: May 11, 2019
# Analyze Reddit data using text analysis and text mining techniques.
# Downloaded 2007 data here: https://files.pushshift.io/reddit/comments/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_profiling as pp
import random
import datetime
from datetime import datetime, date, time

import nltk
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk import FreqDist


#reddit_01_2019 = pd.read_table(r'/Users/Tanner/Documents/python/reddit/RC_2009-01')

#reddit_01_2019.head()

#reddit_01_2007 = pd.DataFrame()
reddit_01_2007 = pd.read_json(r'/Users/Tanner/Documents/python/reddit/RC_2007-01',lines=True)

#f = open('/Users/Tanner/Documents/python/reddit/RC_2009-01', 'r')
#for line in f:
#    df = pd.concat([reddit_01_2019, pd.read_json(line, orient='columns')])
#    

reddit_01_2007.columns
profile = pp.ProfileReport(reddit_01_2007)

profile
profile.to_file(outputfile=r'/Users/Tanner/Documents/python/reddit/profile.html')


#aaron = reddit_01_2007.loc[reddit_01_2007['author'] == 'AaronSw']
#aaron

# We care most about the "body", perhaps "controversiality" 
# Could be a target to train? predicting this based on language?
# we also have "edited", "score", and "subreddit" to make use of
# can we create a flag for post vs. comment?
#Data dictionary:
#https://github.com/reddit-archive/reddit/wiki/JSON

#need to better understand "id", "link id", and "parent id"


#1. Find top words of each subreddit 
reddit_01_2007['body_words'] = reddit_01_2007['body'].apply(lambda x: x.split(' '))
reddit_01_2007['num_body_words'] = reddit_01_2007['body'].apply(lambda x: len(x.split(' ')))
reddit_01_2007['num_unique_body_words'] = reddit_01_2007['body'].apply(lambda x: len(set(x.split(' '))))
reddit_01_2007['num_body_punctuation'] = reddit_01_2007['body'].apply(lambda x: sum(x.count(y) for y in ',:;.'))

reddit_01_2007['num_body_words'].head()
reddit_01_2007['num_unique_body_words'].head()
reddit_01_2007['num_body_punctuation'].head()

#show counts for each score, sort by most frequent scores using the text and other options
#IDEA: could predict score based on inital text? 
#    could predict + or -? +10+ or -10-? multi-class?
reddit_01_2007['score'].value_counts().sort_values


reddit_01_2007['score'].std()
reddit_01_2007.groupby(['subreddit'])['score'].mean().sort_values(ascending=False)
reddit_01_2007.groupby(['subreddit'])['score'].std().sort_values(ascending=False)

#2. top words associated high positive score (top 10%) and negative score (bottom 10%)
reddit_01_2007['score_pct'] = reddit_01_2007['score'].rank() / len(reddit_01_2007['score'])

#Show sample
#reddit_01_2007['score_pct'].sample(20)

#Check for NAs -> 0 so far
reddit_01_2007[reddit_01_2007['score_pct'].isna() == True]

#Graph Scores
#score_plot = np.histogram(reddit_01_2007['score_pct'],bins=10)
plt.hist(reddit_01_2007['score'],bins=30)
plt.xlim(-75,75)
plt.show()

plt.hist(reddit_01_2007['score_pct'],bins=30)
#plt.xlim(-75,75)
plt.show()
#score_plot[:,1]

#12.66% are negative scores 0
len(reddit_01_2007[reddit_01_2007['score'] < 0])/len(reddit_01_2007)
#11% are scores of 0
len(reddit_01_2007[reddit_01_2007['score'] == 0])/len(reddit_01_2007)
#29.3% are scores of 1
len(reddit_01_2007[reddit_01_2007['score'] == 1])/len(reddit_01_2007)
#46% have at least 2 upvotes
len(reddit_01_2007[reddit_01_2007['score'] > 1])/len(reddit_01_2007)



def top_10_pct(pct):
    if pct >= .9:
        return 1
    else:
        return 0
    
def bottom_10_pct(pct):
    if pct <= .1:
        return 1
    else: 
        return 0


reddit_01_2007['top_10_pct'] = reddit_01_2007['score_pct'].apply(top_10_pct)

reddit_01_2007['bottom_10_pct'] = reddit_01_2007['score_pct'].apply(bottom_10_pct)


#show some averages for top middle and bottom

print("Top 10 pct average word count:"
,round(reddit_01_2007.loc[reddit_01_2007['top_10_pct'] == 1]['num_body_words'].mean(),2)
,'\n'
,"Bottom 10 pct average word count:"
,round(reddit_01_2007.loc[reddit_01_2007['bottom_10_pct'] == 1]['num_body_words'].mean(),2)
,'\n'
,"Middle 80 pct average word count:"
,round(reddit_01_2007.loc[(reddit_01_2007['bottom_10_pct'] == 0) & (reddit_01_2007['top_10_pct'] == 0)]['num_body_words'].mean(),2)
,'\n'
)

print("Top 10 pct average count of unique words:"
,round(reddit_01_2007.loc[reddit_01_2007['top_10_pct'] == 1]['num_unique_body_words'].mean(),2)
,'\n'
,"Bottom 10 pct average count of unique words:"
,round(reddit_01_2007.loc[reddit_01_2007['bottom_10_pct'] == 1]['num_unique_body_words'].mean(),2)
,'\n'
,"Middle 80 pct average count of unique words:"
,round(reddit_01_2007.loc[(reddit_01_2007['bottom_10_pct'] == 0) & (reddit_01_2007['top_10_pct'] == 0)]['num_unique_body_words'].mean(),2)
,'\n'
)

print("Top 10 pct average punctuation count:"
,round(reddit_01_2007.loc[reddit_01_2007['top_10_pct'] == 1]['num_body_punctuation'].mean(),2)
,'\n'
,"Bottom 10 pct average punctuation count:"
,round(reddit_01_2007.loc[reddit_01_2007['bottom_10_pct'] == 1]['num_body_punctuation'].mean(),2)
,'\n'
,"Middle 80 pct average punctuation count:"
,round(reddit_01_2007.loc[(reddit_01_2007['bottom_10_pct'] == 0) & (reddit_01_2007['top_10_pct'] == 0)]['num_body_punctuation'].mean(),2)
,'\n'
)


#Define Stopwords
stopwords = nltk.corpus.stopwords.words('english')

#use body or body_words?
body_top_10_pct = reddit_01_2007.loc[reddit_01_2007['top_10_pct'] == 1,'body']
body_bottom_10_pct = reddit_01_2007.loc[reddit_01_2007['bottom_10_pct'] == 1,'body']
body_middle_80_pct = reddit_01_2007.loc[(reddit_01_2007['top_10_pct'] == 0) & 
                                        (reddit_01_2007['bottom_10_pct'] == 0),'body']


#tokens = word_tokenize(reddit_01_2007['body_words'].to_string())
#tokens = word_tokenize(reddit_01_2007['body'].str.cat(sep= ' '))

allWords_top_10_pct = nltk.tokenize.word_tokenize(body_top_10_pct.str.cat(sep=' '))
allWords_bottom_10_pct = nltk.tokenize.word_tokenize(body_bottom_10_pct.str.cat(sep=' '))
allWords_middle_80_pct = nltk.tokenize.word_tokenize(body_middle_80_pct.str.cat(sep=' '))


#allWords = nltk.RegexpTokenizer(r'\w+').tokenize(body_top_10_pct.str.cat(sep=' '))
#allWordDist = FreqDist([w.lower() for w in allWords])

#allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords) 

common_words = pd.DataFrame()
common_words_top_10_pct = nltk.FreqDist(w.lower() for w in allWords_top_10_pct if (w.isalpha()) & (w.lower() not in stopwords))
common_words['top'] = common_words_top_10_pct.most_common()[0:40]

common_words_bottom_10_pct = nltk.FreqDist(w.lower() for w in allWords_bottom_10_pct if (w.isalpha()) & (w.lower() not in stopwords))
common_words['bottom'] = common_words_bottom_10_pct.most_common()[0:40]

common_words_middle_80_pct = nltk.FreqDist(w.lower() for w in allWords_middle_80_pct if (w.isalpha()) & (w.lower() not in stopwords))
common_words['middle'] = common_words_middle_80_pct.most_common()[0:40]
common_words


#check stopword list
check_stopwords = ['a','the','will','testest']
    
for x in check_stopwords:
    if x in stopwords:
        print(x)

    
#3. time of day posted and relation to avg score
#reddit_01_2007['created_utc']

reddit_01_2007['created_dt'] = reddit_01_2007['created_utc'].apply(lambda x: datetime.fromtimestamp(x))
reddit_01_2007['created_dt_time'] = reddit_01_2007['created_dt'].apply(lambda x: datetime.strftime(x,"%H:%M:%S"))
reddit_01_2007['created_hour'] = reddit_01_2007['created_dt'].apply(lambda x: datetime.strftime(x,"%H")).astype(int)

def created_time_of_day(time):
    if time < 6:
        return 'overnight'
    if (time >= 6) & (time < 12):
        return 'morning'
    if (time >= 12) & (time < 18):
        return 'afternoon'
    if time >= 18:
        return 'evening'
    
reddit_01_2007['created_time_of_day']  = reddit_01_2007['created_hour'].apply(created_time_of_day)


#morning has the highest score, while evening has the lowest. early bird gets the worm?
reddit_01_2007.groupby(['created_time_of_day'])['score'].mean()
reddit_01_2007.groupby(['created_time_of_day'])['score'].std()


allWords_morning = nltk.tokenize.word_tokenize(reddit_01_2007.loc[reddit_01_2007['created_time_of_day'] == 'morning', 'body'].str.cat(sep=' '))
allWords_afternoon = nltk.tokenize.word_tokenize(reddit_01_2007.loc[reddit_01_2007['created_time_of_day'] == 'afternoon', 'body'].str.cat(sep=' '))
allWords_evening = nltk.tokenize.word_tokenize(reddit_01_2007.loc[reddit_01_2007['created_time_of_day'] == 'evening', 'body'].str.cat(sep=' '))
allWords_overnight = nltk.tokenize.word_tokenize(reddit_01_2007.loc[reddit_01_2007['created_time_of_day'] == 'overnight', 'body'].str.cat(sep=' '))

common_words_time = pd.DataFrame()

common_words_time_morning = nltk.FreqDist(w.lower() for w in allWords_morning if (w.isalpha()) & (w.lower() not in stopwords))
common_words_time['morning'] = common_words_time_morning.most_common()[0:40]

common_words_time_afternoon = nltk.FreqDist(w.lower() for w in allWords_afternoon if (w.isalpha()) & (w.lower() not in stopwords))
common_words_time['afternoon'] = common_words_time_afternoon.most_common()[0:40]

common_words_time_evening = nltk.FreqDist(w.lower() for w in allWords_evening if (w.isalpha()) & (w.lower() not in stopwords))
common_words_time['evening'] = common_words_time_evening.most_common()[0:40]

common_words_time_overnight = nltk.FreqDist(w.lower() for w in allWords_overnight if (w.isalpha()) & (w.lower() not in stopwords))
common_words_time['overnight'] = common_words_time_overnight.most_common()[0:40]
common_words_time


#4. Score each subreddit and post with pre-trained word embeddings for sentiment
#    , could take avg of the whole subreddit as well

#5. Train a word2vec (or gloVe / spaCy) unique to reddit?

#6. toxicworldcloud of results with Reddit logo for data visualization?




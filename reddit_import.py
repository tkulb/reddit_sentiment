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
import nltk
from nltk.corpus import stopwords

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

reddit_01_2007['num_body_words'].head()
reddit_01_2007['num_unique_body_words'].head()

#show counts for each score, sort by most frequent scores using the text and other options
#IDEA: could predict score based on inital text? 
#    could predict + or -? +10+ or -10-? multi-class?
reddit_01_2007['score'].value_counts().sort_values


reddit_01_2007.groupby(['subreddit'])['score'].mean().sort_values(ascending=False)
reddit_01_2007.groupby(['subreddit'])['score'].std().sort_values(ascending=False)

#2. top words associated high positive score (top 5%) and negative score (bottom 5%)
reddit_01_2007['score_pct'] = reddit_01_2007['score'].rank() / len(reddit_01_2007['score_pct'])

#Show sample
#reddit_01_2007['score_pct'].sample(20)

#Check for NAs -> 0 so far
reddit_01_2007[reddit_01_2007['score_pct'].isna() == True]

#Graph Scores
score_plot = np.histogram(reddit_01_2007['score_pct'],bins=10)
plt.hist(score_plot)
score_plot[:,1]


def top_5_pct(pct):
    if pct >= .95:
        return 1
    else:
        return 0
    
def bottom_5_pct(pct):
    if pct <= .05:
        return 1
    else: 
        return 0

#3. time of day posted and relation to avg score

#4. Score each subreddit and post with pre-trained word embeddings for sentiment
#    , could take avg of the whole subreddit as well

#5. Train a word2vec (or gloVe / spaCy) unique to reddit?

#6. toxicworldcloud of results with Reddit logo for data visualization?




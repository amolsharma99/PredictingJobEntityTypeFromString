import pandas as pd
import numpy as np
import json
import os
import tqdm
import re
import xgboost as xgb
from sklearn import preprocessing
from sklearn import cross_validation 

def extract_date(row):
    return row['postedDate']['$date']

def get_df_from_json(filename, mode):    
    with open(filename) as json_data:
        data = json_data.readlines()
        data = map(lambda x: x.rstrip(), data)
        data_json_str = "[" + ','.join(data) + "]"
        df = pd.read_json(data_json_str)
        if mode != 'test':
            df = df.drop('_id', axis = 1)
            df.postedDate = df.apply(lambda row: extract_date(row), axis = 1)
        return df

df = get_df_from_json('jobs_huge.json', mode = 'train')

new_df = pd.DataFrame(columns=['string', 'class'])
columns = df.columns

#wanted to name it extract
def extra_string_n_class(row, new_df):
    for column in tqdm.tqdm(columns):
        new_df.loc[len(new_df)]=[row[column], column] 
        

df.apply(lambda row: extra_string_n_class(row, new_df), axis = 1)

# to save time if my ipython starts again.
new_df.to_csv('transformed_jobs.csv', encoding='utf-8', index=False)
new_df = pd.read_csv('transformed_jobs.csv', encoding='utf-8')

######Feature Engineering#######
def all_digits(key):
    try:
        x = int(key)
    except ValueError: 
        return 0
    return 1

def num_digits(key):
    try:
        count = sum(c.isdigit() for c in key)
        return count
    except TypeError:
        print "error while counting digts in", key
        return 10

def has_html_tags(key):
    try:
        pat = re.compile('<.*?>')
        match = re.match(pat, key)
        if match:
            return 1
        else:
            return 0
    except TypeError:
        print "error while has_html_tags  in", key
        return 10
    
def len_str(key):
    return len(key)

def occurance_count(df, key, keyname):
    return len(df[df[keyname] == key])

#save occurance as feature and then drop duplicates
new_df['occurance_count'] = new_df.apply(lambda row: occurance_count(new_df, row['string'], 'string'), axis = 1)
new_df = new_df.drop_duplicates()
# New feature columns 'all_digits', 'num_digits', 'has_html_tags', 'len_str', 'is_known_country', 'occurance_count'
new_df['all_digits'] = new_df.apply(lambda row: all_digits(row['string']), axis = 1)
new_df['num_digits'] = new_df.apply(lambda row: num_digits(row['string']), axis = 1)
new_df['has_html_tags'] = new_df.apply(lambda row: has_html_tags(row['string']), axis = 1)
new_df['len_str'] = new_df.apply(lambda row: len_str(row['string']), axis = 1)
###########Classification############
le_class = preprocessing.LabelEncoder()
le_class.fit(new_df['class'])
print le_class.classes_
new_df['en_class'] = le_class.transform(new_df['class'])

Y = new_df.en_class
X = new_df.drop(['string','class', 'en_class'], axis = 1)
Y = Y.reshape(6048, 1)

clf = xgb.XGBClassifier(objective='reg:logistic', nthread=4, seed=0)  
clf.fit(X,Y)

Y = Y.reshape(6048,)
#by default 3 fold cross_validation
scores = cross_validation.cross_val_score(clf, X, Y)
print "3 fold scores: ", scores

print "training set score: ", clf.score(X,Y)
#accurcy 99% on training set

test_df = get_df_from_json('test/test_tiny_1.txt', mode = 'test')

test_new_df = test_df[test_df['key']!='']
#5112 non-empty keys, 882 empty keys.

# to save time if my ipython starts again.
test_new_df.to_csv('transformed_test.csv', encoding='utf-8', index=False)
test_new_df = pd.read_csv('transformed_test.csv', encoding='utf-8')

test_new_df['occurance_count'] = test_new_df.apply(lambda row: occurance_count(test_new_df, row['key'], 'key'), axis = 1)
test_new_df = test_new_df.drop_duplicates()

strings = test_new_df['key']
# New feature columns 'all_digits', 'num_digits', 'has_html_tags', 'len_str', 'is_known_country', 'occurance_count'
test_new_df['all_digits'] = test_new_df.apply(lambda row: all_digits(row['key']), axis = 1)
test_new_df['num_digits'] = test_new_df.apply(lambda row: num_digits(row['key']), axis = 1)
test_new_df['has_html_tags'] = test_new_df.apply(lambda row: has_html_tags(row['key']), axis = 1)
test_new_df['len_str'] = test_new_df.apply(lambda row: len_str(row['key']), axis = 1)

id = test_new_df.id
X = test_new_df.drop(['actual', 'id', 'key'], axis = 1)
Y_predict = clf.predict(X)
Y_predict = le_class.inverse_transform(Y_predict)
#dropped empty keys and dropped duplicates.
print len(id), len(Y_predict), len(strings)

ans_df = pd.DataFrame({'id': id, 'actual': Y_predict, 'key': strings})
ans_df.to_csv('test_tiny_1_out.csv', index= False, encoding='utf=8')
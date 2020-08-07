# sentisum topic sentiment detection

# dataset preparation

import numpy as np
import pandas as pd 
import re
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

data = pd.read_csv('dataset.csv')
def isNaN(string):
    return string != string
#print(data['description'].values[:5])

def preprocess_text(text):
    sentence_words = ''
    text = re.sub('[^a-zA-Zа-яА-Я0-9]+', ' ', text)
    # convert to lowercase
    text = text.lower()
    # remove special characters
    text = re.sub("(?i)[.@#]", "", text)
    # removing addition and subtraction signs
    text = re.sub(' +', ' ', text)
    text = re.sub(' -', ' ', text)
    # removing numbers
    text = re.sub('[0-9]', '', text)
    # removing splitwords

    for words in text.split(' '):
        if(words in stopwords):
            continue
        else:
            sentence_words+=words
            sentence_words+=' '

    return sentence_words


print(data.head())


# total number of classes
# number of columns with topics in the dataset are 14 
class_list = list()
for i in range(1,14):
    series = data['topic_'+ str(i)]
    series = series.dropna()
    for items in list(series):
        class_list.append(items)

# to determine the number of unique sentiments (conversion into set)
myset = set(class_list)
print(list(myset))

# determine the number of each sentiment 
class_dist = {}
for items in list(myset):
    class_dist.update({items:0})
for k,v in class_dist.items():
    for items in list(myset):
        if (items==k):
            class_dist.update({k:v+class_list.count(items)})
   
columns_list = list(myset)
columns_list.insert(0,'text')
#print(class_dist)

dataframe = pd.DataFrame(columns = columns_list)
test_dataframe = pd.DataFrame(columns = columns_list)
val_dataframe = pd.DataFrame(columns = columns_list)

# splitting the dataset in 80:20
train_split = 0.8
print(type(data.index))
print(data.values.shape[0])
j = 0

data['description_good'] = data['description'].apply(preprocess_text)
print(data.description_good)
# determine the examples with no labels
num_no_label = 0

for ind in data.index: 
    j+=1
    print(j)
    #print(df['Name'][ind], df['Stream'][ind]) 
    topic_dict = {}
    description = data['description_good'][ind]
    #print(description)
    
    topic_dict.update({'text':description})
    for items in list(myset):
        topic_dict.update({items:0})
    no_labels = True
    # if a particular example does not have a label, move it to the test set
    for i in range(1,14):
        # check if the string is NaN
        if (not isNaN(data['topic_'+str(i)][ind])):
            no_labels = False
            #print(data['topic_'+str(i)][ind])
            topic_dict.update({data['topic_'+str(i)][ind]:1})
        
        else :
            continue
    if (no_labels==False):
        
        num_no_label+=1
        #print(topic_dict['text'])
        if (j<=data.values.shape[0]*train_split):
            dataframe = dataframe.append(topic_dict,ignore_index = True)
        else:
            test_dataframe = test_dataframe.append(topic_dict,ignore_index = True)

    if (no_labels == True):
        val_dataframe = val_dataframe.append(topic_dict,ignore_index = True)
    
print(num_no_label)

print(dataframe.head())
dataframe.to_csv('train.csv')
test_dataframe.to_csv('val.csv')
val_dataframe.to_csv('test.csv')


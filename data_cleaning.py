import csv
import pandas as pd
import numpy as np
import re
import math
import string
import nltk

from string import punctuation
from nltk.stem import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords

def strip_all_entities(text): 
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower()
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) 
    text = re.sub(r'[^\x00-\x7f]',r'', text) 
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

def find_label(content):

    labels=[]
    for i in range(0,len(content)):
        lst = content[i][0]
        if lst=='-':
        	labels.append(-1)
        elif lst=='0':
        	labels.append(0)
        else:
        	labels.append(1)
    return labels





def isnotNaN(string):
    return string == string




def depure_data(content,lenght):
	for i in range(0,lenght):
		lst=[]
		if isnotNaN(content[i]):
			data=content[i]
			lst.append(data)
		else:
			data='-'
			lst.append(data)
	return lst



def preprocess_text(content,lenght):
	lst=[]
	for z in range(0,lenght):
		text=content[z]
		stemmer= PorterStemmer()
		if isnotNaN(text):
			text = re.sub(r"(?:\@|https?\://)\S+", "", text) 
			text = re.sub(r'[^\x00-\x7f]',r'', text)
			text = re.sub('\S*@\S*\s?', '', text)
			text = re.sub('\s+', ' ', text)
			text = re.sub("\'", "", text)
			text = text.lower()
			text = re.sub(f"[{re.escape(punctuation)}]", "", text)
			text = ''.join([j for j in text if not j.isdigit()])
			text = remove_stopwords(text)
			text=stemmer.stem(text)
			text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() 
			banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
			table = str.maketrans('', '', banned_list)
			text = text.translate(table)
			text=re.sub("\s\s+" , " ", text)
			text = " ".join(text.split())
			lst.append(text)
		else:
			text=='nothing'
			lst.append(text)

	return lst



#read_data
data = pd.read_csv('semeval-2017-test.csv', error_bad_lines=False)
l=len(data['label	text'])


df = pd.DataFrame(data={"Label":find_label(data['label	text']),"Sentence_A":preprocess_text(data['label	text'],l),"Sentence_B":preprocess_text(data['B'],l),"Sentence_C":preprocess_text(data['C'],l),"Sentence_D":preprocess_text(data['D'],l),"Sentence_E":preprocess_text(data['E'],l)})
df.to_csv("./output.csv", sep=',',index=False)


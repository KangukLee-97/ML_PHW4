from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords



#----------------------document----------------------
sentences = list()
with open("cran.all.1400.txt",encoding="UTF-8") as file:
    line=file.readlines()
for l in line:
    sentences.append(l.rstrip())
joined_str = " ".join(sentences)

new_sentences = list()

keywords=[".T",".I",".A",".B"]

for l in re.split(".W|.I", joined_str):
    if any(keyword in l for keyword in keywords):
        continue
    else:
        new_sentences.append(l)

document_df=pd.DataFrame({"contents":new_sentences})

print(document_df)


#----------------------query----------------------
qry = list()
with open("cran.qry.txt",encoding="UTF-8") as file:
    line=file.readlines()
for l in line:
    qry.append(l.rstrip())
joined_qry = " ".join(qry)

#----------------------make DataFrame for Query----------------------

qry_sentences = list()
for l in re.split(".I", joined_qry):
    if l:
        l=re.split(".W",l)
        qry_sentences.append(l[1])
joined_qry = " ".join(qry_sentences)
qry_df=pd.DataFrame({"contents":qry_sentences})

print(qry_df)

#----------------------preprocessing for Query----------------------
punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
for ele in joined_qry:
    if ele in punc:
        joined_qry = joined_qry.replace(ele, " ")
joined_qry=joined_qry.lower()

for i in range(1):
    # this will convert
    # the word into tokens
    text_tokens = word_tokenize(joined_qry)

tokens_without_sw = [
    word for word in text_tokens if not word in stopwords.words()]

print(tokens_without_sw)


#----------------------inverted index----------------------
dict = {}
for i in range(len(new_sentences)):
    check = new_sentences[i].lower()
    for item in tokens_without_sw:
        if item in check:
            if item not in dict:
                dict[item] = []

            if item in dict:
                if i+1 in dict[item]:
                    continue
                else:
                    dict[item].append(i + 1)


#----------------------Data Frame of inverted index----------------------

k_list = list(dict)
df1 = pd.DataFrame(k_list)
v_list = list(dict.values())
df2 = pd.DataFrame([v_list]).transpose()
s_list = []
for i in range(len(v_list)):
  s_list.append(len(v_list[i]))
df3 = pd.DataFrame(s_list)

df = pd.concat([df1, df2, df3], axis = 1)
df.columns = ["tokens", "documentNum","size"]

#----------------------Handling frequent tokens----------------------
indexNames = df[(df["size"]>=500)].index
df.drop(indexNames, inplace=True)

print(df)

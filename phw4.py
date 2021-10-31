from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
import pandas as pd
import re
import random

from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords


# ---------------------- document ----------------------
sentences = []
with open("cran/cran.all.1400.txt", encoding="UTF-8") as file:
    line = file.readlines()
    for l in line:
        sentences.append(l.rstrip())
    joined_str = " ".join(sentences)

    new_sentences = []
    keywords = [".T", ".I", ".A", ".B"]

    for l in re.split(".W|.I", joined_str):
        if any(keyword in l for keyword in keywords):
            continue
        else:
            new_sentences.append(l)

    document_df = pd.DataFrame({"contents": new_sentences})

    print(document_df)

    # ---------------------- query ----------------------
    qry = []
    with open("cran/cran.qry.txt", encoding="UTF-8") as file:
        line = file.readlines()
        for l in line:
            qry.append(l.rstrip())
    joined_qry = " ".join(qry)

    # ---------------------- Spliting & preprocessing Query ----------------------
    qry_sentences = []
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''

    for l in re.split(".I", joined_qry):
        if l:
            l = re.split(".W", l)
            for ele in l[1]:
                if ele in punc:
                    l[1] = l[1].replace(ele, " ")
            qry_sentences.append(l[1])
    joined_qry = " ".join(qry_sentences)
    joined_qry = joined_qry.lower()

    # ----------------------make DataFrame for Query ----------------------
    qry_df = pd.DataFrame({"contents": qry_sentences})
    print(qry_df)

    # ---------------------- tokenizing query ----------------------
    for i in range(1):
        # this will convert
        # the word into tokens
        text_tokens = word_tokenize(joined_qry)

    tokens_without_sw = [
        word for word in text_tokens if not word in stopwords.words()]

    # ---------------------- inverted index ----------------------
    dict = {}
    for i in range(len(new_sentences)):   # new_sentences: document
        check = new_sentences[i].lower()
        for item in tokens_without_sw:
            if item in check:
                if item not in dict:
                    dict[item] = []

                if item in dict:
                    if i in dict[item]:
                        continue
                    else:
                        dict[item].append(i)

    # ---------------------- Data Frame of inverted index ----------------------
    k_list = list(dict)
    df1 = pd.DataFrame(k_list)
    v_list = list(dict.values())
    df2 = pd.DataFrame([v_list]).transpose()
    s_list = []
    for i in range(len(v_list)):
        s_list.append(len(v_list[i]))
    df3 = pd.DataFrame(s_list)

    inverted_index_df = pd.concat([df1, df2, df3], axis=1)
    inverted_index_df.columns = ["tokens", "documentNum", "size"]

    # ---------------------- Handling frequent tokens ----------------------
    indexNames = inverted_index_df[(inverted_index_df["size"] >= 500)].index
    inverted_index_df.drop(indexNames, inplace=True)
    inverted_index_df = inverted_index_df.sort_values(by="tokens", ascending=True)
    print(inverted_index_df)

    # ---------------------- Query test ----------------------
    query_index_list = []
    for i in range(5):
        ranNum = random.randint(0, qry_df.shape[0]-1)   # 0부터 224까지
        query_index_list.append(ranNum)

    for q_idx in query_index_list:
        print("********** Query {0} **********".format(q_idx))
        example = qry_df.iloc[q_idx, 0]
        for i in range(1):
            text_tokens = word_tokenize(example)
        example_tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

        # ---------------------- Query tokenizer ----------------------
        document_list = []
        for word in example_tokens_without_sw:
            for i in inverted_index_df["tokens"].values:
                if i == word:
                    document_list.append(inverted_index_df[inverted_index_df.tokens == word].iloc[0, 1])

        print(document_list)  # Query token이 들어가있는 document 번호들 list

        # ---------------------- get document in inverted index ----------------------
        doc_list = []
        for i in range(len(document_list)):
            for j in range(len(document_list[i])):
                doc_list.append(document_list[i][j])
        ex1_document_df = document_df.loc[doc_list]
        print(ex1_document_df)  # query token이 들어가있는 document 번호들 list를 dataframe으로 (document 번호 + 내용들)

        # ---------------------- caclulate cosine similarity ----------------------
        total_list = ex1_document_df['contents'].values.tolist()
        total_list.append(example)  # query token이 들어가있는 document 번호들 list를 dataframe으로 한 것에 마지막에 query 내용 추가
        cvec = TfidfVectorizer(stop_words='english')
        dc = cvec.fit_transform(total_list)
        cosine_sim = cosine_similarity(dc, dc)
        print(cosine_sim)

        # ----------------------make dataframe ----------------------
        ex1_document_df.insert(loc=1, column="cosine", value=cosine_sim[len(ex1_document_df), 0:-1])
        ex1_document_df.sort_values("cosine", ascending=False, inplace=True)
        ex1_document_df.drop_duplicates(inplace=True)
        print(ex1_document_df.head(50))

        # ----------------------The matrix of cranqrel----------------------
        cranqrel = pd.DataFrame()
        # column_Num=[]
        for i in range(1, 1401):
            cranqrel.insert(i - 1, i, 0)
        print(cranqrel)

        with open("cran/cranqrel.txt", encoding="UTF-8") as file:
            line = file.readlines()
        for l in line:
            str = re.findall("\d+", l)
            if (str[0] in cranqrel.index):
                cranqrel.loc["{}".format(str[0]), int(str[1])] = 1.0
            else:
                cranqrel.loc[str[0]] = 0.0
                cranqrel.loc["{}".format(str[0]), int(str[1])] = 1.0
        cranqrel = cranqrel.fillna(0.0)
        print(cranqrel)

        # ----------------------The matrix of example query----------------------
        ex1_cf = pd.DataFrame()
        for i in range(1, 1401):
            ex1_cf.insert(i - 1, i, 0)
        print(ex1_cf)

        n = [10, 15, 20]
        cs = [0.2, 0.1, 0.08]

        ex1_result = []
        for num in n:
            for e in cs:
                temp = ex1_document_df.iloc[0:num]
                t = temp.copy()
                t.drop(temp[temp["cosine"] >= e].index, inplace=True)
                ex1_result.append(temp.index.to_list())

        for li in ex1_result:
            ex1_cf.loc[0] = 0.0
            for n in li:
                ex1_cf.loc[0, n] = 1.0
            print(classification_report(cranqrel.iloc[q_idx+1, :].values, ex1_cf.iloc[0, :].values, digits=4))
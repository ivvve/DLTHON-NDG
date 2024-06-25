import os
import pandas as pd
from konlpy.tag import Okt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Mecab
import numpy as np
from collections import Counter
import re


# 형태소 분석하는 함수 
def morph_analyze(text):
    
    # Okt 형태소 분석기 객체 생성
    okt = Okt()
    # 텍스트에서 형태소 분석
    morphs = okt.morphs(text)
    return morphs


# 각 문장의 불용어, 특수 문자 등을 제거하는 function: {return: 정리된 문장}
def sentence_analysis(sentence):
    sentence = re.sub(r'@[^@]+@', 'pronoun', sentence)
    sentence = re.sub(r'name1', 'pronoun', sentence)
    sentence = re.sub(r'company-name' , 'pronoun', sentence)
    
    # 특수문자 제거 (문장내의 특수 문자제거)
    sentence = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z?.!\s]", "", sentence)
    
    # 영어라면 소문자로 변환
    sentence = sentence.lower() # 텍스트 소문자화
    #형태소 분석
    sentence = morph_analyze(sentence)
    
    # 불용어 제거 
    sen = []
    for word in sentence:
        if word in stopwords:
            continue
        sen.append(word)
       
    sentence = ' '.join(sen)

    return sentence



#단어 사전 만들어주는 function: {토큰: index} 와 {index: 토큰} dictionary를 제공
def makeVocab(train, train_check = True):
    words = []
    
    for sentence in train['conversation']:
        temp = list(string.split(" "))
        words.extend(temp)
        
    counter = Counter(words)
    counter = counter.most_common(10000-4)
    vacab
    
    vocab = ['', '', '', ''] + [key for key, _ in counter]
    word_to_index = {word:index for index, word in enumerate(vocab)}
    #실제 인코딩 인덱스는 제공된 word_to_index에서 index 기준으로 3씩 뒤로 밀려 있습니다.  
    word_to_index = {k:(v+3) for k,v in word_to_index.items()}

    # 처음 몇 개 인덱스는 사전에 정의되어 있습니다.
    word_to_index["<PAD>"] = 0
    word_to_index["<BOS>"] = 1
    word_to_index["<UNK>"] = 2  # unknown
    word_to_index["<UNUSED>"] = 3

    index_to_word = {index:word for word, index in word_to_index.items()}
    return word_to_index, index_to_word




#token화된 list를 정수화로 바꿔주는 function : {return: 정수화된 list}
def wordlist_to_indexlist(wordlist, word_to_index):
        return [word_to_index[word] if word in word_to_index else word_to_index[''] for word in wordlist]
    
    

# 결측치 제거, 중복 제거, 불용어 제거한 데이터를 제공하는 fucntion
def load_data(path):
    train_data_path = path
    data = pd.read_csv(train_data_path)
    
    # 결측치 제거
    null_check = data.isnull().sum()
    check = False
    for i in range(len(null_check)):
        if null_check[i] > 0:
            check = True
            
    if check == True:
        data = data.dropna()
    
    # 중복 제거
    data.drop_duplicates(subset = ['conversation'], inplace=True)
    
    #불용어
    stopwords = ['은','는','이','가','을','를','에','이가','이는']
    
    data = data['conversation'].map(lambda x: sentence_analysis(x))

    return data


# class 협박 대화, 갈취 대화, 직장 내 괴롭힘 대화, 기타 괴롭힘 대화, 일반 대화를 0,1,2,3,4로 바꾸는 function
def changeClassInt(data):
    # class_list = {'협박 대화': 0, '갈취 대화': 1. '직장 내 괴롭힘 대화': 2, '기타 괴롭힘 대화': 3, '일반 대화': 4}
    data.loc[data['class'] == '협박 대화', 'class'] = 0
    data.loc[data['class'] == '갈취 대화', 'class'] = 1
    data.loc[data['class'] == '직장 내 괴롭힘 대화', 'class'] = 2
    data.loc[data['class'] == '기타 괴롭힘 대화', 'class'] = 3
    data.loc[data['class'] == '일반 대화', 'class'] = 4
    
    return data


# 데이터 분리 function
def makeDataset(cov_data, tar_data):
    # stratify : class가 균등하게 나눠지게 됨.train_test_split stratify
    
    X_train, X_val, y_train,y_val = train_test_split(cov_data, tar_data, test_size = 0.2, random_state = 928, stratify = y_data)    
    
    return X_train, X_val, y_train, y_val 




# 문장 1개를 활용할 딕셔너리와 함께 주면, 단어 인덱스 리스트 벡터로 변환해 주는 함수입니다. 
# 단, 모든 문장은 <BOS>로 시작하는 것으로 합니다. 
def get_encoded_sentence(sentence, word_to_index):
    return [word_to_index['<BOS>']]+[word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in sentence.split()]

# 여러 개의 문장 리스트를 한꺼번에 단어 인덱스 리스트 벡터로 encode해 주는 함수입니다. 
def get_encoded_sentences(sentences, word_to_index):
    return [get_encoded_sentence(sentence, word_to_index) for sentence in sentences]

# 숫자 벡터로 encode된 문장을 원래대로 decode하는 함수입니다. 
def get_decoded_sentence(encoded_sentence, index_to_word):
    return ' '.join(index_to_word[index] if index in index_to_word else '<UNK>' for index in encoded_sentence[1:])  #[1:]를 통해 <BOS>를 제외

# 여러 개의 숫자 벡터로 encode된 문장을 한꺼번에 원래대로 decode하는 함수입니다. 
def get_decoded_sentences(encoded_sentences, index_to_word):
    return [get_decoded_sentence(encoded_sentence, index_to_word) for encoded_sentence in encoded_sentences]


#importing library
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


total_data=pd.read_table('shopping.txt',names=['ratings','reviews']) #reading data as a table and name columns as ratings and reviews
total_data.head() # cheching just 5lines of data

total_data['label']=np.select([total_data.ratings>3],[1],default=0)  #Dividing dating into two class :0,1 , reviews by ratings above 3 will count as 1 and have positive sentiment
#and ratings with values of below 3 will have class 0 so negative sentiment

total_data.head()

total_data.drop_duplicates(subset=['reviews'],inplace=True) # drop duplicates if we have some

len(total_data)

from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(total_data ,test_size=0.25, random_state=111)  # 75% of data will be use as traing data and the rest testing data
print(len(train_data))
print(len(test_data))

train_data['label'].value_counts().plot(kind='bar')  #visualization label column of training data and we can see we have almost amount of posive sentence as
# negative sentence which is good for training our data

print(train_data.groupby('label').size().reset_index(name='count'))

train_data['reviews']=train_data['reviews'].str.replace("[^ㄱ-ㅎ ㅏ-ㅣ 가-힣 ]","")  #removing all non-korean charactor from reviews 
train_data['reviews'].replace('',np.nan,inplace=True)  #replace all empty stirngs in columns with NaN value
print(len(train_data))

test_data['reviews']=test_data['reviews'].str.replace("[^ㄱ-ㅎ ㅏ-ㅣ 가-힣 ]","") #also do same thing for testing data
test_data['reviews'].replace('',np.nan,inplace=True)
print(len(test_data))

stopwords=['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']
# stopwors is a list of non useful words in Korean language for analyzing data

from eunjeon import Mecab #Mecab is a morphological analyzer for korean language
mecab=Mecab()

train_data['reviews']=train_data['reviews'].apply(mecab.morphs) #tokenize our reviews data by applying mecab library for reviews column
train_data['reviews']=train_data['reviews'].apply(lambda x:[item for item in x if item not in stopwords]) #remove words which are same as stopwords list

test_data['reviews']=test_data['reviews'].apply(mecab.morphs)
test_data['reviews']=test_data['reviews'].apply(lambda x:[item for item in x if item not in stopwords])

#checking amount and shape of our data againg
X_train=train_data['reviews'].values
y_train=train_data['label'].values
X_test=test_data['reviews'].values
y_test=test_data['label'].values

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)  #learn data and then encode the text to numerical data


threshold=3  #number of occurrences for a word to be considered "rare"
words_cnt=len(tokenizer.word_index)
rare_cnt=0
words_freq=0
rare_freq=0

for key,value in tokenizer.word_counts.items():
    words_freq=words_freq + value
    
    if value < threshold:
        rare_cnt +=1
        rare_freq=rare_freq + value
        
print("전체 단어 수:",words_cnt)
print("빈도가 {} 이하인 희귀 단어 수:{}".format(threshold-1,rare_cnt))
print("회귀 단어 비율: {}".format((rare_cnt / words_cnt)*100))
print("희귀 단어 등장 빈도 비용: {}".format((rare_freq / words_freq)*100))

vocab_size=words_cnt-rare_cnt+2
print(vocab_size)

tokenizer=Tokenizer(vocab_size,oov_token='oov')
tokenizer.fit_on_texts(X_train)
X_train=tokenizer.texts_to_sequences(X_train)
X_test=tokenizer.texts_to_sequences(X_test)

X_train[:2]

print(X_test[:2])

plt.hist([len(s)for s in X_train],bins=50)
plt.xlabel("lengh of samples")
plt.ylabel("number of samples")
plt.show()

max_len=60
X_train=pad_sequences(X_train,maxlen=max_len)
X_test=pad_sequences(X_test,maxlen=max_len)

from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding,Dense,GRU
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

model=Sequential()
model.add(Embedding(vocab_size,100))
model.add(GRU(128))
model.add(Dense(1,activation='sigmoid'))

es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=4)
mc=ModelCheckpoint('b_model.h5',monitor='val_acc',mode='max',verbose=1,save_best_only=True)

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(X_train,y_train,epochs=15,callbacks=[es,mc],batch_size=60,validation_split=0.2)

load_model=load_model('b_model.h5')
load_model.evaluate(X_test,y_test)

test=pd.read_csv('기말과제.csv')
test=test['reviews']

answer=[]
def sentiment_predict(sentence):
    for new_sentence in sentence:
        
        new_token=[word for word in mecab.morphs(new_sentence) if not word in stopwords]
        new_sequences=tokenizer.texts_to_sequences([new_token])
        new_pad=pad_sequences(new_sequences,maxlen=max_len)
        score=float(load_model.predict(new_pad))
    
        if score>0.5:
            answer.append(1)
        else:
            answer.append(0)

sentiment_predict(test)




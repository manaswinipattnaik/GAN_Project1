import pandas as pd
import numpy as np
import re
import nltk
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer


#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.stem import wordnet
from nltk.corpus import stopwords

lemma=WordNetLemmatizer()
#apply pandas to read data and NLP for preprocessing

class Sentiment:
    def __init__(self):
        
        self.df=pd.read_csv('C:\\Users\\debas\\Desktop\\machine_learning_projects\\sentiment_analysis\\Reviews_amazon.csv')
        #print(self.df.columns)
        self.df1=self.df.head(1000)
        self.df1=self.df1.loc[:,['Text','Score']]
        #print(self.df1)
        self.df1['Text']=self.df1['Text'].apply(self.modify)
        #print(self.df1['Text'])

    
        def f(x):
            if x==4 or x==5:
                return 1
            elif x==3:
                return 2
            else:
                return 0
    
        self.df1['Score']=self.df1['Score'].apply(f)
        print(self.df1)

        self.review=self.df1['Text'].values
        self.label=self.df1['Score'].values

        self.voca_size=10000;self.max_length=100
        self.token=Tokenizer(num_words=self.voca_size,oov_token='<00V>')
        self.token.fit_on_texts(self.review)
        self.seq=self.token.texts_to_sequences(self.review)
        self.pad_seq=pad_sequences(self.seq,maxlen=self.max_length,padding='post')
        
        self.model=Sequential([Embedding(self.voca_size,64,input_length=self.max_length),
                   LSTM(64,dropout=0.2),
                  Dense(3,activation='softmax')
                  
                  ]
                 )
    
        self.model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.pad_seq,self.label,test_size=0.2,random_state=8)
        self.model.fit(self.x_train,self.y_train,epochs=50,batch_size=32,validation_data=(self.x_test,self.y_test))
        self.loss,self.acc=self.model.evaluate(self.x_test,self.y_test)  
        print(self.acc)
        self.model.summary()
        print(type(self.x_test))
        self.lis=[]
        pred=self.model.predict(self.x_test)
        for i in pred:
            self.lis.append(np.argmax((np.array(i))))
       
        print(self.lis)

        #inference
        review1='i like product'
        review1=self.modify(review1)
        review1=self.token.texts_to_sequences([review1])
        review1=pad_sequences(review1,maxlen=self.max_length,padding='post')
        pred1=self.model.predict(review1)
        print(pred1)


    def modify(self,x):
        x=re.sub(r'\W',' ',x)
        x=re.sub(r'\s+[a-zA-Z]\s',' ',x)
        x=re.sub(r'\^[a-zA-z]\s+',' ',x)
        x=re.sub(r'\s+',' ',x)
        x=x.lower()
        x=x.split()
        x=[lemma.lemmatize(w,pos='v') for w in x ]
        x=[w for w in x if w not in stopwords.words('english')]
        x=' '.join(x)
        return x



if __name__=='__main__':
    ob=Sentiment()
    







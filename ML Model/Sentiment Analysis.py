#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis

# In[1]:


# importing the necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data = pd.read_csv('Downloads/Google Play Store Apps Dataset/googleplaystore_user_reviews.csv')


# In[3]:


data.head(5)


# We need only the Translated_Review and Sentiment columns. Lets get it.

# In[4]:


df = pd.concat([data['Translated_Review'],data['Sentiment']],axis=1)
df.shape


# In[5]:


#Checking the null values.
df.isna().sum()


# In[6]:


#Dropping the null values.
df.dropna(axis=0,inplace=True)


# In[7]:


df.shape


# Our target "Sentiment" is of 3 types : Positive, Negative, Neutral

# In[8]:


df['Sentiment'].value_counts()


# 
# We convert it to the our desired form

# In[9]:


df.Sentiment=[0 if i=="Positive" else 1 if i== "Negative" else 2 for i in df.Sentiment]


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(data.Sentiment)
plt.title("Count of Sentiments")


# ## Data Preprocessing

# We use NLP to pre-process the data
# Natural language processing (NLP) is a subfield of computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data.
# 
# Our steps are:
# 
# Tokenization
# 
# Stopwords
# 
# Lemmatazation/Stemming
# 
# Vectorization

# In[11]:


df.Translated_Review[0]


# 
# The values in the Translated_Review column has some characters which are not letters. They need to be removed.Let's try for a first row

# In[12]:


import re
first_text=df.Translated_Review[0]
text=re.sub("[^a-zA-Z]"," ",first_text) #replacing every character other than alphabets, with space
text=text.lower()


# text now contains a value which contains only letters.

# In[13]:


text


# 
# Tokenization is done below.
# 
# Tokenization splits the sentence into words.
# 
# We then try to identify the stopwords.
# 
# stopwords - words like "the" which are not useful and only make our list of words bigger.
# We remove these stopwords.

# In[14]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
# in anaconda prompt: python -m nltk.downloader all
from nltk.corpus import stopwords 
text=nltk.word_tokenize(text) # word_tokenize-separate all words


# In[15]:


len(text)


# In[16]:


text


# Identifying Stop Words Next, we want to consider the importance of a each word in the sentence. English has a lot of filler words that appear very frequently like “and”, “the”, and “a”. When doing statistics on text, these words introduce a lot of noise since they appear way more frequently than other words. Some NLP pipelines will flag them as stop words —that is, words that you might want to filter out before doing any statistical analysis.

# In[17]:


text = [w for w in text if not w in stopwords.words('english')]
len(text)


# Lemmatization- converts a word to its root form.

# In[18]:


import nltk as nlp
nltk.download('wordnet')
lemma=nlp.WordNetLemmatizer()
text=[lemma.lemmatize(i) for i in text]
text=" ".join(text)
text


# Extending the above steps for the entire dataset

# In[19]:


text_list=[]
for i in df.Translated_Review:
    text=re.sub("[^a-zA-Z]"," ",i)
    text=text.lower()
    text=nltk.word_tokenize(text)
    text = [w for w in text if not w in stopwords.words('english')]
    lemma=nlp.WordNetLemmatizer()
    text=[lemma.lemmatize(word) for word in text]
    text=" ".join(text)
    text_list.append(text)


# In[20]:


len(text)


# Vectorization:
# 
# Here we turn a collection of text documents into numerical feature vectors.
# From vectorization we can get a dataframe with the each column representing a word from the bag of words and each row represents a review.So if a word appears in a review then it's count is the value in that row corresponding to that column.

# In[21]:



from sklearn.feature_extraction.text import CountVectorizer
max_features=200000  # a random value that we specify as the size of the bag of words
cou_vec=CountVectorizer(max_features=max_features,stop_words="english")
sparce_matrix=cou_vec.fit_transform(text_list).toarray() # vectorization
all_words=cou_vec.get_feature_names() # bag of words
print(np.amax(sparce_matrix))


# In[22]:



words = pd.DataFrame(sparce_matrix,columns=all_words)


# In[ ]:


from wordcloud import WordCloud
plt.subplots(figsize=(12,12))
wordcloud=WordCloud(background_color="white",width=1024,height=768).generate(" ".join(all_words[4500:]))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[ ]:


len(all_words)


# In[ ]:


y=df.iloc[:,1].values
x=sparce_matrix
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[ ]:


x


# In[ ]:


y


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
print("lr accuracy: ",lr.score(x_test,y_test))


# In[ ]:


#confussion matrix
y_pred=lr.predict(x_test)
y_true=y_test
from sklearn.metrics import confusion_matrix
import seaborn as sns
names=["Positive","Negative","Neutral"]
cm=confusion_matrix(y_true,y_pred)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=.5,linecolor="r",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# ## Neural Networks

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:



model = Sequential([
    Dense(3, activation='relu',input_shape=(x_train[1].shape)),    # input_shape should be given only 1d
    Dense(3, activation='softmax')
])

model.summary()


# In[ ]:


adam = keras.optimizers.adam(lr=0.001)

model.compile(loss='sparse_categorical_crossentropy',
             optimizer = adam,
              metrics = ['accuracy']
             )


# In[ ]:


history = model.fit(x_train,y_train,epochs = 5, verbose = 1)


# In[ ]:


testing = model.evaluate(x_test,y_test,verbose=1)


# In[ ]:


testing


# In[ ]:


model.metrics_names


# In[ ]:





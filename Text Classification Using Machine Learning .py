#!/usr/bin/env python
# coding: utf-8

# # **Text Classification Using Machine Learning **

# # Importing Libraries

# 

# In[8]:


# Importing the necessary libraries for data manipulation and visualisation
import numpy as np  # NumPy is a Python library used for working with arrays.
import pandas as pd  # Pandas is used for data manipulation and analysis.
import matplotlib.pyplot as plt  # Matplotlib is a plotting library for creating static, animated, and interactive visualisations in Python.
import seaborn as sns  # Seaborn is a Python data visualisation library based on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
import warnings  # This is a module for issuing warning messages as part of your program.

warnings.warn('Ignore')
# plt.style.use('dark_background'
sns.set_style("dark")
print('Done')


# # Loading Dataset

# In[9]:


#loading dataset using pandas library
df = pd.read_csv('Dataset.csv', encoding='latin-1')


# In[11]:


#displaying first 10 records of the dataset 
df.tail(20)


# In[12]:


#Displaying the shape of the datset, means how many columns and rows we have in dataset
df.shape


# In[13]:


#shows us the info about the dataset about the datatypes 
df.info()


# # Data Preprocessing

# ## Data Cleaning

# In[14]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True,axis=1)


# In[15]:


df.head(10)


# In[16]:


df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)


# In[17]:


df.columns


# In[18]:


df.head(5)


# In[19]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target']=encoder.fit_transform(df['target'])


# In[20]:


df.isnull().sum()


# In[21]:


df.duplicated().sum


# In[22]:


df=df.drop_duplicates(keep='first') # deletes the first occurrence.


# In[23]:


df.shape


# # Exploratory Data Analysis

# In[24]:


df['target'].value_counts()


# In[25]:


plt.pie(df['target'].value_counts(), labels=['ham','spam'], autopct='%0.2f',colors = ['#ff9999','#66b3ff'])
plt.show()


# #### Data is imbalanced

# In[26]:


# Natural Language Toolkit
import nltk


# In[27]:


nltk.download('punkt')


# #### This tokenizer divides a text into a list of sentences by using anunsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences. It must be trained on a large collection of plaintext in the target language before it can be used.

# ## Fetch number of characters

# In[29]:


df['num_chars']=df['text'].apply(len)


# In[30]:


df.head(8)


# ## Fetch number of words

# In[31]:


df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[32]:


df.head(8)


# ## Number of sentences

# In[33]:


df['num_sentence']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[34]:


df.head()


# ## For ham messages

# In[35]:


df[['num_chars','num_words','num_sentence']].describe()


# In[36]:


df[df['target']==0][['num_chars','num_words','num_sentence']].describe()


# ## For spam messages

# In[37]:


df[df['target']==1][['num_chars','num_words','num_sentence']].describe()


# In[38]:


plt.figure(figsize=(14,5))
sns.histplot(data=df,x='num_chars',hue="target",palette="inferno",kde=True); 


# In[39]:


plt.figure(figsize=(14,5))
sns.histplot(data = df,hue='target',x='num_words',palette="inferno", kde=True); 


# In[40]:


plt.figure(figsize=(14,5))
sns.pairplot(df,hue='target',palette='inferno');


# In[41]:


sns.heatmap(df.corr(),annot=True);


# # NLP Data Preprocessing
# - Lower case
# - Tokenization
# - Removing special characters
# - Removing stop words and punctuation
# - Stemming

# In[42]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from nltk.corpus import stopwords
import string


# In[43]:


def TextTransform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


# In[45]:


import nltk
nltk.download('stopwords')
stopwords.words("english") 


# ### These words  do no give any meaning to the sentence but are used in sentence formation

# In[134]:


# string.punctuation


# In[46]:


TextTransform('It is Spamming spammed spam Detection! 20%%')


# In[47]:


df['text'][0]


# In[48]:


df['transformed_text'] = df['text'].apply(TextTransform)


# In[49]:


df.head(20)


# In[51]:


from wordcloud import WordCloud
wc= WordCloud(width=500,height=500, min_font_size = 10, background_color='black')


# In[52]:


spam_wordcloud = wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))


# In[53]:


plt.figure(figsize=(14,5))
plt.imshow(spam_wordcloud)


# In[54]:


ham_wordcloud = wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))


# In[55]:


plt.figure(figsize=(14,5))
plt.imshow(ham_wordcloud)


# ### Top 30 spam messages

# In[56]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[57]:


len(spam_corpus)


# In[58]:


df.info()


# In[59]:


from collections import Counter
most_common_messages = pd.DataFrame(Counter(spam_corpus).most_common(30))
most_common_messages.columns = ["words","freq"]
plt.figure(figsize=(14,5))
sns.barplot(data = most_common_messages,y="words",x="freq", palette = 'inferno');
plt.title("Most Common Spam Messages")
# plt.xticks(rotations='vertical')


# ### Top 30 ham messages

# In[60]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[61]:


len(ham_corpus)


# In[62]:


from collections import Counter
most_common_ham = pd.DataFrame(Counter(ham_corpus).most_common(30))
most_common_ham.columns = ["Words","Frequency"]
plt.figure(figsize=(14,5))
sns.barplot(data = most_common_ham,y="Words",x="Frequency", palette = 'inferno');
plt.title("Most Common Ham Messages")
# plt.xticks(rotations='vertical')


# # Model Building

# ### Text Vectorization using Bag of Words

# In[63]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[64]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[65]:


X.shape


# In[66]:


y = df['target'].values
y


# In[67]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[69]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, classification_report


# In[70]:


gnb_model = GaussianNB()
gnb_model.fit(X_train,y_train)
y_predict = gnb_model.predict(X_test)
print("="*100)
print("Accuracy Score:",accuracy_score(y_test,y_predict))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_predict))
print("\n",classification_report(y_test,y_predict))


# In[71]:


mnb_model = MultinomialNB()
mnb_model.fit(X_train,y_train)
y_predict = mnb_model.predict(X_test)
print("="*100)
print("Accuracy Score:",accuracy_score(y_test,y_predict))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_predict))
print("\n",classification_report(y_test,y_predict))


# In[72]:


bnb_model = BernoulliNB()
bnb_model.fit(X_train,y_train)
y_predict = bnb_model.predict(X_test)
print("="*100)
print("Accuracy Score:",accuracy_score(y_test,y_predict))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_predict))
print("\n",classification_report(y_test,y_predict))


# In[74]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[75]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[76]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


# In[77]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[78]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[79]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[80]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)
performance_df


# In[189]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")
performance_df1


# In[190]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[191]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)
new_df = performance_df.merge(temp_df,on='Algorithm')
new_df_scaled = new_df.merge(temp_df,on='Algorithm')
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)
new_df_scaled.merge(temp_df,on='Algorithm')


# ## Voting Classifier

# In[200]:


svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')
voting.fit(X_train,y_train)
mnb.fit(X_train,y_train)


# In[201]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# ## Applying Stacking

# In[202]:


estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()
from sklearn.ensemble import StackingClassifier
clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[204]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


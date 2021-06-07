import pandas as pd
df=pd.read_csv("/content/Restaurant_Reviews.tsv" ,delimiter='\t',quoting=3)
import re # Regular Expressions
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] 
for i in range(0,1000):
  review = re.sub('[^a-zA-Z]',' ', df['Review'][i] )
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords=stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
  from sklearn.feature_extraction.text import CountVectorizer
  cv=CountVectorizer(max_features=1500)

x=cv.fit_transform(corpus).toarray()
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2 , random_state=0)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix , accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)

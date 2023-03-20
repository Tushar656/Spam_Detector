import pandas as pd
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
from nltk.corpus import wordnet
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# from pyscript import Element


stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


# <---------------------  READ DATASET  -------------------------->
dataset = pd.read_csv('./spam.csv', encoding='latin-1')
x = dataset['v1']
y = dataset['v2']
# print(y)
length = y.size


# <---------------------  LOWER CASE    --------------------------->
for i in range (length):
    y[i] = y[i].lower()


# <------------------  REMOVE STOP WORDS  ------------------------->
y = y.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
# print(dataset['tweet_without_stopwords'])


# <-----------------  REMOVE PUNCTUATIONS  ------------------------>
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
for i in range (length):
    y[i] = remove_punctuation(y[i])

# print(y[0])




# <-------------------   LEMMITAZATION   ------------------------->
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
for i in range (length):
    y[i] = lemmatize_words(y[i])

# print(y[0])






le = LabelEncoder()
x = le.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(y, x, test_size=0.2, random_state=1)


tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

tfidfRg = LogisticRegression()
tfidfRg.fit(X_train_tfidf, y_train)
y_pred_tfidf = tfidfRg.predict(X_test_tfidf)



def spamModel(test):
    # input_mail = ["free entri 2 wkli comp win fa cup final tkt 21st may 2005 text fa 87121 receiv entri questionstd txt ratetc appli 08452810075over18"]
    input_mail = [test]

    input_data_features = tfidf_vectorizer.transform(input_mail)

    prediction = tfidfRg.predict(input_data_features)
    return prediction

print(spamModel("Congratulations! You won 1 Lakh rupees on your mobile number, to recieve money call to 9161286870"))






# my_innerHtml = Element('out')
# my_innerHtml.write(prediction)




# words = nltk.word_tokenize(y[0])
# pos_tags = nltk.pos_tag(words)
# print(pos_tags)

# print(dataset['tweet_without_stopwords'].unique())

# print(stop)
# print(lemmatizer)
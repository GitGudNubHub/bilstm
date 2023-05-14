import pandas as pd
import numpy as np
import re
from keras.callbacks import ModelCheckpoint
import math
from keras import regularizers
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.preprocessing import MinMaxScaler
from keras.utils import pad_sequences
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional, Attention
from keras.models import Model
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Concatenate, Input, Embedding, BatchNormalization
from keras.preprocessing.text import Tokenizer
import time
from tqdm.keras import TqdmCallback
from keras.callbacks import Callback, ProgbarLogger
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('wordnet')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re

tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\S+', '', text)
    # Tokenize text using TweetTokenizer
    tokens = tokenizer.tokenize(text)
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token, pos=wordnet.VERB) for token in tokens]
    # Join tokens back into a string
    text = ' '.join(tokens)
    return text

# Load the dataset into a Pandas dataframe, this datasaet is from kaggle sentiment140, but i removed the neutral sentiments, which had a value of 2. Here we convert the 4s, which are positive to 1.
data = pd.read_csv('LR_Sentiment.csv')


# Convert the label column to binary (0 for negative sentiment, 1 for positive sentiment)
data['label'] = data['label'].replace({0: 0, 4: 1})

# apply the preprocessing

data['content'] = data['content'].apply(preprocess_text)
print(data.head())

#Tokenize the text data using the bag-of-words technique
vectorizer = CountVectorizer( strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', stop_words='english',max_features=12000)
data = data.dropna()
X = vectorizer.fit_transform(data['content'])


vocab_size = len(vectorizer.vocabulary_) + 1
print(vocab_size)

max_length = 255
padded_sequences = pad_sequences(X.toarray(), maxlen=max_length)
print(vocab_size)



X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data['label'], test_size=0.20, random_state=98)


model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=255))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()



early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model with early stopping
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64,callbacks=[early_stopping])

# Test the model on new data
# Test the model on new data

y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

X_test_reshaped = np.squeeze(X_test)  # Remove the last dimension with size 1
X_test_flat = X_test_reshaped.reshape((-1,))

# Evaluate the model using accuracy, precision, recall, and F1-score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(acc*100))
print("Precision: {:.2f}%".format(prec*100))
print("Recall: {:.2f}%".format(rec*100))
print("F1-Score: {:.2f}%".format(f1*100))
model.save('reducedSentimentBOW_model1.h5')



# create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# plot the accuracy curves
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# plot the loss curves
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()



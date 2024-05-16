import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

# Load the dataset
df = pd.read_csv("C:\\Users\\HP\\Downloads\\email spam\\emails.csv")

# Visualize spam distribution with a donut chart
spam_counts = df['spam'].value_counts()
plt.figure(figsize=(8, 6))
colors = ['skyblue', 'orange']
plt.pie(spam_counts, labels=['Not Spam', 'Spam'], autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.4))
plt.title('Spam Distribution')
plt.show()

# Prepare data
X = df['text'].astype(str)
y = df['spam'].replace({0: "Not Spam", 1: "Spam"}).astype("object")

# Plot histogram of email lengths
email_lengths = df['text'].apply(len)
plt.figure(figsize=(8, 6))
sns.histplot(email_lengths, bins=50, kde=False, color='blue')
plt.title('Histogram of Email Lengths')
plt.xlabel('Email Length')
plt.ylabel('Frequency')
plt.show()

# Plot KDE plot of email lengths vs. spam
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df, x=email_lengths, hue='spam', fill=True, palette=['skyblue', 'orange'])
plt.title('KDE Plot of Email Lengths vs. Spam')
plt.xlabel('Email Length')
plt.ylabel('Density')
plt.show()

# Train and evaluate Naive Bayes classifier
class NaiveBayes:
    def __init__(self):
        self.log_prior = {}
        self.word_counts = {}
        self.total_words = {}

    def fit(self, X, y):
        # Calculate the log prior probabilities
        classes, counts = np.unique(y, return_counts=True)
        total_docs = len(y)
        for cls, count in zip(classes, counts):
            self.log_prior[cls] = np.log(count / total_docs)
            self.word_counts[cls] = defaultdict(int)
            self.total_words[cls] = 0
        
        # Calculate word counts and total words for each class
        for doc, label in zip(X, y):
            for word in doc.split():
                self.word_counts[label][word] += 1
                self.total_words[label] += 1

        # Calculate the log likelihoods
        self.log_likelihood = {}
        for cls in classes:
            self.log_likelihood[cls] = {}
            total_words_cls = self.total_words[cls]
            for word, count in self.word_counts[cls].items():
                self.log_likelihood[cls][word] = np.log((count + 1) / (total_words_cls + len(self.word_counts[cls])))

    def predict(self, X):
        predictions = []
        for doc in X:
            posterior = {}
            for cls in self.log_prior:
                posterior[cls] = self.log_prior[cls]
                for word in doc.split():
                    if word in self.log_likelihood[cls]:
                        posterior[cls] += self.log_likelihood[cls][word]
            predictions.append(max(posterior, key=posterior.get))
        return predictions




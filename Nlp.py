# Importing necessary libraries
import spacy
import nltk
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from gensim import corpora
from gensim.models import LdaModel

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# Function to clean text data
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Function for Tokenization and Lemmatization
def tokenize_and_lemmatize(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return tokens

# Sentiment Analysis using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Returns sentiment polarity (-1 to 1)

# Named Entity Recognition (NER) using spaCy
def named_entity_recognition(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Text Classification using TF-IDF and Naive Bayes
def text_classification(texts, labels):
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(classification_report(y_test, predictions))

# Topic Modeling with LDA (Latent Dirichlet Allocation)
def topic_modeling(texts, num_topics=5):
    # Preprocessing and Tokenization
    texts_clean = [tokenize_and_lemmatize(clean_text(text)) for text in texts]
    
    # Creating a dictionary and corpus
    dictionary = corpora.Dictionary(texts_clean)
    corpus = [dictionary.doc2bow(text) for text in texts_clean]
    
    # Training LDA model
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = lda.print_topics(num_words=5)
    
    for topic in topics:
        print(topic)

# Example Data (Text Data)
texts = [
    "I love programming in Python. It's such a versatile language!",
    "The weather today is really bad. It's too cold outside.",
    "I enjoy reading books about Artificial Intelligence and Data Science.",
    "I feel very happy today! It's such a great day.",
    "I'm excited about the new technological advancements in NLP."
]

labels = ['positive', 'negative', 'positive', 'positive', 'positive']  # Example labels for classification

# Clean and preprocess the text
cleaned_texts = [clean_text(text) for text in texts]
tokenized_texts = [tokenize_and_lemmatize(text) for text in cleaned_texts]

# 1. Sentiment Analysis
for text in texts:
    sentiment = analyze_sentiment(text)
    print(f"Sentiment of '{text}': {sentiment}")

# 2. Named Entity Recognition (NER)
for text in texts:
    entities = named_entity_recognition(text)
    print(f"Named entities in '{text}': {entities}")

# 3. Text Classification
print("Text Classification:")
text_classification(texts, labels)

# 4. Topic Modeling
print("Topic Modeling:")
topic_modeling(texts, num_topics=2)

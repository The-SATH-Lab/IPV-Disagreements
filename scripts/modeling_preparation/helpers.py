import pandas as pd
import numpy as np
from datetime import datetime
import re
import string
import nltk
import emoji
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

slang_mapping = {
    "u": "you", "r": "are", "gr8": "great", "omg": "oh my god", "ain't": "am not",
    "aren't": "are not", "can't": "cannot", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it's": "it is", "it'll": "it will", "let's": "let us", 
    "might've": "might have", "must've": "must have", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "that's": "that is", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have", "wasn't": "was not",
    "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what'll": "what will", "what're": "what are", "what's": "what is",
    "what've": "what have", "where's": "where is", "who'd": "who would", "who'll": "who will",
    "who're": "who are", "who's": "who is", "who've": "who have", "won't": "will not",
    "would've": "would have", "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
    "you're": "you are", "you've": "you have"
}

def load_data(file_path):
    """Loads data from the given file path."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)

def aggregate_embeddings(group_df):
    """Aggregates embeddings by taking the mean."""
    embeddings = np.vstack(group_df['Embedding'].values)
    return pd.Series(np.mean(embeddings, axis=0))

def calculate_average_response_time(filtered_data):
    """Calculates average response time for each participant per day."""
    response_time_data = {
        "Participant ID": [],
        "Date": [],
        "AvgResponseTime": []
    }

    participants = filtered_data['Participant ID'].unique()

    for participant in participants:
        participant_data = filtered_data[filtered_data['Participant ID'] == participant]
        
        for date in participant_data['Date'].unique():
            date_data = participant_data[participant_data['Date'] == date]
            initial_action = date_data.iloc[0]['Message_Action']

            sent_times = []
            received_times = []
            time_differences = []

            if initial_action == 'sent':
                sent_times.append(date_data.iloc[0]['Time'])
            else:
                received_times.append(date_data.iloc[0]['Time'])            

            for _, row in date_data.iterrows():
                if initial_action == 'received' and row['Message_Action'] == 'sent':
                    sent_times.append(row['Time'])
                    initial_action = 'sent'
                elif initial_action == 'sent' and row['Message_Action'] == 'received':
                    received_times.append(row['Time'])
                    initial_action = 'received'

            if sent_times and received_times:
                for sent_time, received_time in zip(sent_times, received_times):
                    try:
                        time1 = datetime.combine(date, sent_time)
                        time2 = datetime.combine(date, received_time)
                        time_difference = time1 - time2
                        time_diff_seconds = time_difference.total_seconds()
                        time_differences.append(int(time_diff_seconds))
                    except ValueError as e:
                        print(f"Error parsing time: {e}")
                        continue

            if time_differences:
                avg_response_time = int(sum(time_differences) / len(time_differences))
                response_time_data['Participant ID'].append(participant)
                response_time_data['Date'].append(date)
                response_time_data['AvgResponseTime'].append(avg_response_time)

    return pd.DataFrame(response_time_data)

def clean_text(text):
    """Cleans and preprocesses the text for sentiment analysis."""
    if isinstance(text, (int, float)):
        return str(text)  # Convert numeric types to string
    text = re.sub(r'^Thug Life: ', '', text)
    text = text.lower()
    text = contractions.fix(text)
    text = text.replace("'", "")
    text = re.sub('\[.*?\]', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    text = re.sub('\w*\d\w*', '', text)
    text = emoji.demojize(text)  # Convert emojis to text
    text = re.sub('_', ' ', text)  # Replace underscores with space
    text = text.replace(":", " ")  # Remove colons from demojized text
    text = " ".join([slang_mapping.get(word, word) for word in text.split()])  # Translate slang
    words = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)

def analyze_sentiment(text):
    """Analyzes the sentiment of the text."""
    if isinstance(text, (int, float)):
        return 0
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity < 0:
        return -1
    return 0

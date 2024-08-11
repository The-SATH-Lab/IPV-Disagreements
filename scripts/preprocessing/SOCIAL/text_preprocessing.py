import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from scripts.config import RAW_SOCIAL_MESSAGING_DIR, PROCESSED_SOCIAL_MESSAGING_DIR
import pickle

def load_data(path):
    """Loads data from the given file path."""
    if path.endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_excel(path)

def preprocess_social_messaging_data():
    new_columns = ['Message_Action', 'Text', 'Relationship', 'Application']
    dfs = []

    # Load and preprocess data
    for path in sorted(os.listdir(RAW_SOCIAL_MESSAGING_DIR)):
        if path.endswith(('.xlsx', '.csv')):
            try:
                df = load_data(os.path.join(RAW_SOCIAL_MESSAGING_DIR, path))
                df.columns = [df.columns[0]] + new_columns
                df['Relationship'] = df['Relationship'].astype(str)
                df.rename(columns={df.columns[0]: path.split('.')[0]}, inplace=True)
                dfs.append(df)
            except Exception as e:
                continue

    # Calculate frequency of relationships
    labels = set()
    freq = defaultdict(int)

    for df in dfs:
        current_labels = df['Relationship'].unique()
        labels.update(current_labels)
        for label in current_labels:
            freq[label] += (df['Relationship'] == label).sum()

    freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)}

    # Labels for classification
    labels_to_consider = ['Romantic Partner', 'Important Person', 'Parental Figure', 'Friend']
    labels_greater_200 = [label for label in freq.keys() if label not in labels_to_consider and freq[label] > 200]

    def classify_relationship(x):
        """Classifies the relationship based on the text."""
        x = x.lower()
        if any(keyword in x for keyword in ['romantic partner', 'partner', 'romantic']):
            return 'Romantic Partner'
        if 'friend' in x or 'important person' in x:
            return 'Friend'
        if any(keyword in x for keyword in ['father', 'mother', 'parent', 'dad', 'mom']):
            return 'Parental Figure'
        if x in labels_greater_200:
            return 'Friend'
        if x not in labels_to_consider:
            return 'Others'
        return x

    for df in dfs:
        df['Exact_Relationship'] = df['Relationship']
        df['Relationship'] = df['Relationship'].apply(classify_relationship)

    # Adding message type features
    message_type = defaultdict(int)
    for df in dfs:
        for text in df['Text']:
            if isinstance(text, str) and text.isupper():
                message_type[text] += 1

    message_type = {k: v for k, v in sorted(message_type.items(), key=lambda item: item[1], reverse=True)}

    def classify_message_type(x):
        """Classifies the message type based on its content."""
        if isinstance(x, str):
            if 'VIDEO' in x:
                return 'VIDEO'
            if 'GIF' in x or 'IMAGE' in x:
                return 'IMAGE'
            if 'ATTACHMENT' in x:
                return 'ATTACHMENT'
        return 'TEXT'

    for df in dfs:
        df['Type'] = df['Text'].apply(classify_message_type)
        df['Word_Count'] = df['Text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], format='ISO8601')
        df = df.convert_dtypes()

    # Filter and clean data
    dfs = [df for df in dfs if 'Romantic Partner' in df['Relationship'].unique()]
    dfs = [df[df['Type'].str.lower() == 'text'] for df in dfs]
    dfs = [df.dropna(subset=['Text']).astype({'Text': 'string'}) for df in dfs]
    dfs = [df for df in dfs if not df.empty]

    # Save processed data
    target_file = os.path.join(PROCESSED_SOCIAL_MESSAGING_DIR, "preprocessed_dfs.pkl")
    with open(target_file, 'wb') as f:
        pickle.dump(dfs, f)

if __name__ == "__main__":
    preprocess_social_messaging_data()

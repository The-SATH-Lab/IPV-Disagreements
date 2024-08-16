import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
from scripts.config import PROCESSED_SNAP_MESSAGING_DIR, PROCESSED_SNAP_EMA_DIR, SAVED_EMBEDDINGS_SNAP_DIR
from .helpers import calculate_average_response_time, clean_text, analyze_sentiment
import warnings
warnings.filterwarnings("ignore")

class SnapProcessor:
    def __init__(self, model_name='distilbert/distilbert-base-uncased', batch_size=32, relationship='Romantic Partner', embedding_method='mean_pooling', save_dir='saved_embeddings'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)
        self.batch_size = batch_size
        self.relationship = relationship
        self.embedding_method = embedding_method
        self.save_dir = save_dir

    def get_embeddings(self, texts):
        """
        Computes embeddings for a list of texts using the specified model and embedding method.
        """
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        if self.embedding_method == 'cls_token':
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Extracting CLS token embedding
        elif self.embedding_method == 'mean_pooling':
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Mean pooling of the last hidden state
        else:
            raise ValueError(f"Unknown embedding method: {self.embedding_method}")

        return embeddings
    
    def batch_apply(self, df, func):
        """Applies a function in batches to the dataframe."""
        num_batches = (len(df) + self.batch_size - 1) // self.batch_size
        embeddings = []

        for i in tqdm(range(num_batches), desc='Processing batches'):
            batch_texts = df['Text'].iloc[i*self.batch_size:(i+1)*self.batch_size].tolist()
            batch_embeddings = func(batch_texts)
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)

    def save_embeddings(self, df):
        """Saves embeddings to a CSV file or loads them if they already exist."""
        model_name_clean = self.model_name.split('/')[-1]  # Extract the last part of the model name

        # Construct the directory structure
        save_path = os.path.join(self.save_dir, model_name_clean, self.relationship.replace(' ', '_'))
        os.makedirs(save_path, exist_ok=True)

        # Filename for the embeddings
        embedding_file = os.path.join(save_path, f"{self.embedding_method}_embeddings.csv")
        
        if not os.path.exists(embedding_file):
            # Compute and save embeddings
            embeddings = self.batch_apply(df, self.get_embeddings)
            
            # Convert embeddings into separate columns
            embedding_df = pd.DataFrame(embeddings, index=df.index)
            embedding_df.columns = [f"Dim_{i}" for i in range(embedding_df.shape[1])]
            
            # Combine with Participant ID and Date
            df = pd.concat([df[['Participant ID', 'Date']], embedding_df], axis=1)
            
            # Save to CSV
            df.to_csv(embedding_file, index=False)
        else:
            print(f"Loading embeddings from {embedding_file}")
            df = pd.read_csv(embedding_file)
            
            # Ensure Date is in datetime format
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        return df

    def snap_processing(self, dataframes):
        """Processes the SNAP messaging data."""
        participant_files = [df.columns[0] for df in dataframes]
        participant_ids = [int(file.split("_")[0]) if file.split("_")[0].isdigit() else int(file.split("_")[1]) for file in participant_files]
        file_to_id_mapping = dict(zip(participant_files, participant_ids))
        
        for df in dataframes:
            df.loc[:, 'Participant ID'] = file_to_id_mapping[df.columns[0]]
            df.rename(columns={df.columns[0]: 'Notification Time'}, inplace=True)
        
        aggregated_df = pd.concat(dataframes, axis=0, ignore_index=True)
        aggregated_df = aggregated_df.reindex(columns=[
            'Participant ID', 'Notification Time', 'Message_Action', 'Text', 'Relationship',
            'Application', 'Exact_Relationship', 'Type', 'Word_Count'
        ])
        aggregated_df.drop_duplicates(inplace=True)
        text_message_df = aggregated_df[aggregated_df['Application'].isin(['textmessage', 'text-messages'])]
        text_message_df = text_message_df.assign(
            Date=text_message_df['Notification Time'].dt.date,
            Time=text_message_df['Notification Time'].dt.time
        )
        text_message_df.drop(['Application', 'Exact_Relationship'], axis=1, inplace=True)
        text_message_df.drop_duplicates(inplace=True)
        
        # Filter by the specified relationships
        filtered_df = text_message_df[text_message_df['Relationship'] == self.relationship]
        return filtered_df

    def process_ema_data(self, ema_dataframe, filtered_dataframe):
        """Processes the EMA data for relevant participants."""
        ema_dataframe.drop_duplicates(inplace=True)
        filtered_ema_df = ema_dataframe.copy()
        filtered_ema_df.loc[:, 'Participant ID'] = filtered_ema_df['Participant ID'].apply(
            lambda x: int(x) if str(x).isdigit() else np.nan
        ).astype('Int64')
        filtered_ema_df = filtered_ema_df[
            filtered_ema_df['Participant ID'].isin(filtered_dataframe['Participant ID'].unique())
        ]
        filtered_ema_df['Notification Time'] = pd.to_datetime(filtered_ema_df['Notification Time'])
        filtered_ema_df['Participant ID'] = filtered_ema_df['Participant ID'].astype('int')
        return filtered_ema_df

    def prepare_modeling_df(self, messaging_dir, ema_dir, embeddings=True, sentiment_analysis=False, message_count=False, return_train_test=False, test_split='random'):
        """
        Prepares the final modeling dataframe, including all necessary processing steps.

        Parameters:
        - messaging_dir (str): Directory containing SNAP messaging data.
        - ema_dir (str): Directory containing EMA data.
        - embeddings (bool): Whether to include text embeddings. Default is True.
        - sentiment_analysis (bool): Whether to include sentiment analysis. Default is False.
        - message_count (bool): Whether to include message count as a feature for the Baseline model. Default is False.
        - return_train_test (bool): Whether to return training and test sets. Default is False.
        - test_split (str): Method to split the data into training and test sets. 
          Accepts 'random' (default) or 'participant_based_split'.

        Returns:
        - modeling_df (DataFrame) or (train_df, test_df) tuple: The final dataframe(s) ready for modeling.
        """
        
        # Load SNAP messaging data
        snap_dfs = pd.read_pickle(os.path.join(messaging_dir, "preprocessed_dfs.pkl"))
        
        # Load EMA data
        ema_dataframe = pd.read_csv(os.path.join(ema_dir, "preprocessed_ema.csv"), parse_dates=['Notification Time', 'Date'])
        
        # Process messaging and EMA data
        filtered_dataframe = self.snap_processing(snap_dfs)
        ema_dataframe = self.process_ema_data(ema_dataframe, filtered_dataframe)
        
        # Ensure that the 'Date' column is in datetime format for both DataFrames
        filtered_dataframe['Date'] = pd.to_datetime(filtered_dataframe['Date'], errors='coerce')
        ema_dataframe['Date'] = pd.to_datetime(ema_dataframe['Date'], errors='coerce')

        # Calculate average response time and word count
        response_time_df = calculate_average_response_time(filtered_dataframe)
        avg_word_count_df = filtered_dataframe.groupby(['Participant ID', 'Date'])['Word_Count'].mean().reset_index()
        avg_word_count_df.rename(columns={'Word_Count': 'AvgWordCount'}, inplace=True)
        
        # Merge response time and word count into a single DataFrame
        aggregated_features_df = pd.merge(response_time_df, avg_word_count_df, on=['Participant ID', 'Date'], how='inner')
        
        # If message_count is True, compute the MessageCount and merge it into the aggregated features
        if message_count:
            message_count_df = filtered_dataframe.groupby(['Participant ID', 'Date']).agg(MessageCount=('Text', 'count')).reset_index()
            aggregated_features_df = pd.merge(aggregated_features_df, message_count_df, on=['Participant ID', 'Date'], how='inner')
        
        # Merge the result with EMA DataFrame
        text_ema_merged_df = pd.merge(aggregated_features_df, ema_dataframe, on=['Participant ID', 'Date'], how='inner', suffixes=('_text', '_ema'))
        
        # If embeddings is True, process and merge embeddings
        if embeddings:
            embeddings_df = self.save_embeddings(filtered_dataframe.copy())  # Use a copy to avoid modifying the original
            embedding_columns = [col for col in embeddings_df.columns if col.startswith('Dim_')]
            
            # Aggregate embeddings
            embeddings_agg_data = embeddings_df.groupby(['Participant ID', 'Date'])[embedding_columns].mean().reset_index()
            
            # Merge aggregated embeddings back into the main DataFrame
            text_ema_merged_df = pd.merge(text_ema_merged_df, embeddings_agg_data, on=['Participant ID', 'Date'], how='inner')
        
        # If sentiment_analysis is True, compute and merge sentiment analysis features
        if sentiment_analysis:
            filtered_dataframe['Cleaned_Text'] = filtered_dataframe['Text'].apply(clean_text)
            filtered_dataframe['Sentiment'] = filtered_dataframe['Cleaned_Text'].apply(analyze_sentiment)
            
            sentiment_counts = filtered_dataframe.groupby(['Participant ID', 'Date', 'Sentiment']).size().unstack(fill_value=0)
            sentiment_counts['NegativeSentimentPercentage'] = sentiment_counts[-1] / (sentiment_counts[-1] + sentiment_counts[1] + sentiment_counts[0])
            sentiment_counts_df = sentiment_counts.reset_index()[['Participant ID', 'Date', 'NegativeSentimentPercentage']]
            
            text_ema_merged_df = pd.merge(text_ema_merged_df, sentiment_counts_df, on=['Participant ID', 'Date'], how='inner')
        
        # Define the feature columns
        feature_columns = ['Participant ID', 'Date', 'AvgResponseTime', 'AvgWordCount']
        if embeddings:
            feature_columns += embedding_columns
        if sentiment_analysis:
            feature_columns.append('NegativeSentimentPercentage')
        if message_count:
            feature_columns.append('MessageCount')
        
        # Group by the necessary columns and process the 'Disagreement' column
        modeling_df = text_ema_merged_df.groupby(['Participant ID', 'Date'] + feature_columns[2:]).agg({
            'Disagreement': 'sum'
        }).reset_index()

        # Convert 'Disagreement' to binary (0 or 1)
        modeling_df['Disagreement'] = modeling_df['Disagreement'].apply(lambda x: 1 if x >= 1 else 0)

        if return_train_test:
            if test_split == 'random':
                snap_train_df, snap_test_df = train_test_split(modeling_df, test_size=0.2,
                                                               stratify=modeling_df['Disagreement'], random_state=42)
            elif test_split == 'participant_based_split':
                snap_train_df = modeling_df[modeling_df['Participant ID'] <= 1120]
                snap_test_df = modeling_df[modeling_df['Participant ID'] > 1120]

            return snap_train_df, snap_test_df

        return modeling_df






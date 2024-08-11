import os
import pandas as pd
from pathlib import Path
from scripts.config import RAW_SNAP_EMA_DIR, PROCESSED_SNAP_EMA_DIR

def process_ema_files(nis_file_name, uis_file_name, mapping_file_name):
    nis_file_path = os.path.join(RAW_SNAP_EMA_DIR, nis_file_name)
    nis_df = pd.read_csv(nis_file_path, encoding='latin-1', low_memory=False)
    nis_df.rename(columns={nis_df.columns[0]: "Participant ID"}, inplace=True)

    uis_file_path = os.path.join(RAW_SNAP_EMA_DIR, uis_file_name)
    uis_df = pd.read_csv(uis_file_path, encoding='latin-1', low_memory=False)
    uis_df.rename(columns={uis_df.columns[0]: "Participant ID"}, inplace=True)

    mapping_file_path = os.path.join(RAW_SNAP_EMA_DIR, mapping_file_name)
    participant_mapping_df = pd.read_csv(mapping_file_path)

    mapping_dict = dict(zip(participant_mapping_df["Participant ID"], participant_mapping_df["IDNO"]))
    nis_df['Participant ID'] = nis_df['Participant ID'].map(mapping_dict)
    uis_df['Participant ID'] = uis_df['Participant ID'].map(mapping_dict)

    ema_df = pd.concat([nis_df, uis_df], axis=0)
    return ema_df

def preprocess_ema_data():
    ema_df_1 = process_ema_files(
        nis_file_name='SNAP_FINAL06.11.20_NISWide_02.14.2024.csv',
        uis_file_name='SNAP_FINAL06.11.20_UISWide_02.14.2024.csv',
        mapping_file_name='SNAP_FINAL06.11.20_StartUp_02.14.2024.csv'
    )

    ema_df_2 = process_ema_files(
        nis_file_name='SNAP_FINAL01.15.20_NISWide_02.25.2021.csv',
        uis_file_name='SNAP_FINAL01.15.20_UISWide_02.25.2021.csv',
        mapping_file_name='SNAP_FINAL01.15.20_StartUp_02.25.2021.csv'
    )
    ema_df_2.rename(columns={'RP_Number (2)': 'RP_Number'}, inplace=True)

    ema_df = pd.concat([ema_df_1, ema_df_2], axis=0)

    ema_df['Participant ID'] = ema_df['Participant ID'].apply(lambda x: int(x.split()[0]) if x.split()[0].isdigit() else x)

    date_columns = ["Notification Time", "Session Start Time"]
    ema_df[date_columns] = ema_df[date_columns].apply(pd.to_datetime)

    ema_df['Notification Time'] = ema_df['Notification Time'].fillna(ema_df['Session Start Time'])
    ema_df.drop('Session Start Time', axis=1, inplace=True)

    responded_mapping = {
        "User didn't respond to this notification": "User didn't respond to this notification",
        '1': 1,
        1: 1,
        0: 0
    }
    ema_df['Responded'] = ema_df['Responded'].map(responded_mapping)

    disagreement_labels = [
        'Rel_Disag1',
        'Disag_2_MI',
        'Disag_3_MI',
        'EOD_Disagr',
        'Disa_Setup',
        'Disa_Setu2'
    ]
    ema_df.dropna(subset=disagreement_labels, how='all', inplace=True)

    exclude_all_99 = ema_df[disagreement_labels].eq(99.0).any(axis=1)
    ema_df = ema_df[~exclude_all_99]

    response_mask = ema_df[disagreement_labels].eq(1).any(axis=1)
    disagreement_series = response_mask.astype(int)

    # Combine the new `Disagreement` column with the existing DataFrame in one operation
    ema_df = pd.concat([ema_df, disagreement_series.rename('Disagreement')], axis=1)

    cols_to_consider = [
        'Participant ID',
        'Notification Time',
        'Disagreement'
    ]
    processed_ema_df = ema_df[cols_to_consider].reset_index(drop=True)

    processed_ema_df['Date'] = processed_ema_df['Notification Time'].dt.date
    processed_ema_df['Time'] = processed_ema_df['Notification Time'].dt.time

    target_file = os.path.join(PROCESSED_SNAP_EMA_DIR, 'preprocessed_ema.csv')
    processed_ema_df.to_csv(target_file, index=False)

if __name__ == "__main__":
    preprocess_ema_data()
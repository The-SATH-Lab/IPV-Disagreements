import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


RAW_SNAP_DIR = os.path.join(DATA_DIR, "raw", "SNAP")
RAW_SNAP_MESSAGING_DIR = os.path.join(RAW_SNAP_DIR, "Messaging Data")
RAW_SNAP_EMA_DIR = os.path.join(RAW_SNAP_DIR, "EMA Data")

RAW_SOCIAL_DIR = os.path.join(DATA_DIR, "raw", "SOCIAL")
RAW_SOCIAL_MESSAGING_DIR = os.path.join(RAW_SOCIAL_DIR, "Messaging Data")
RAW_SOCIAL_EMA_DIR = os.path.join(RAW_SOCIAL_DIR, "EMA Data")



PROCESSED_SNAP_DIR = os.path.join(DATA_DIR, "processed", "SNAP")
PROCESSED_SNAP_MESSAGING_DIR = os.path.join(PROCESSED_SNAP_DIR, "Messaging Data")
PROCESSED_SNAP_EMA_DIR = os.path.join(PROCESSED_SNAP_DIR, "EMA Data")

PROCESSED_SOCIAL_DIR = os.path.join(DATA_DIR, "processed", "SOCIAL")
PROCESSED_SOCIAL_MESSAGING_DIR = os.path.join(PROCESSED_SOCIAL_DIR, "Messaging Data")
PROCESSED_SOCIAL_EMA_DIR = os.path.join(PROCESSED_SOCIAL_DIR, "EMA Data")





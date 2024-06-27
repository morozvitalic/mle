# Importing required libraries
import numpy as np
import pandas as pd
import logging
import os
import sys
import json

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH')

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining default data...")
DATA_SEED = conf['general']['random_state']
INFERENCE_DATA_LEN = conf['inference']['test_sample']

logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])

# Singleton class for generating data set
@singleton
class IrisGenerator():
    def __init__(self):
        self.df = None

    # Method to create the Iris data
    def create_sets(self, test_len: int, train_path: os.path, inference_path: os.path, seed: int = 0):
        logger.info("Creating Iris dataset...")
        from sklearn.datasets import load_iris

        # Load the Iris dataset
        iris = load_iris()

        # Create a DataFrame
        df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
        df['target'] = iris['target']

        # Split the data into training and inference sets
        df_inference = df.sample(n=test_len, random_state=seed)
        df_train = df.drop(df_inference.index)

        # Save the data
        self.save(df_train, train_path)
        self.save(df_inference, inference_path)

    # Method to save data
    def save(self, df: pd.DataFrame, out_path: os.path):
        logger.info(f"Saving data to {out_path}...")
        df.to_csv(out_path, index=False)

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    gen = IrisGenerator()
    gen.create_sets(test_len=INFERENCE_DATA_LEN, train_path=TRAIN_PATH, inference_path=INFERENCE_PATH, seed=DATA_SEED)
    logger.info("Script completed successfully.")
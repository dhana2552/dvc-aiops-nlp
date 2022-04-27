import argparse
from fileinput import filename
import os
import logging
from utils.common import read_yaml, create_directories
import urllib.request as req


STAGE = "stage 01 get data" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    source_data_url = config["source_data_url"]
    # source_data = config["source_data"]
    
    local_data_dir = config["source_download_dirs"]["data_dir"]
    create_directories([local_data_dir])
    
    data_filename = config["source_download_dirs"]["data_file"]
    local_data_filepath = os.path.join(local_data_dir, data_filename)
    
    
    logging.info(f"downloading data from: {source_data_url}")
    filename, headers = req.urlretrieve(source_data_url, local_data_filepath)
    logging.info(f"data downloaded successfully")
    logging.info(f"data saved at: {local_data_filepath}")
    logging.info(f"Downloaded headers: {headers}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
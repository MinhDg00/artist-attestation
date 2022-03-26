from google.cloud import storage
from google.oauth2 import service_account
import os
import glob
import logging

#### Set up authentication if not 
# Note: Please setup services account and create key (which is .json file below)
cur_path = os.getcwd()
credential_path = cur_path + "/agile-scheme-345202-362629af24a2.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

def upload_data(bucket_name: str, source_dir: str, destination_dir: str):
    """Uploads image data to the bucket.

    @bucket_name: Name of Google Cloud Storage
    @source_dir: data path 
    @destination_dir: destination blob in bucket that will receive data
    """

    if not os.path.isdir(source_dir) or not os.path.exists(source_dir):
        raise TypeError(f"{source_dir} is not a directory or does not exist.")

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name, timeout = 24*60*60)

    print(f"UPLOAD STARTED .... \n")

    for source_file_name in glob.glob(source_dir + "/*.jpg"):

        destination_blob_name = destination_dir + "/" + source_file_name.split("/")[-1]
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)

        print(f"File {source_file_name} uploaded to {destination_blob_name}.")

    print(f"UPLOAD SUCCESSFUL!")

# upload_data("cs6384-bucket", "train", "train")
upload_data("cs6384-bucket", "test", "test")


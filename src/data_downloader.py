import csv
import os
import pandas as pd
import requests
import boto3
from botocore import UNSIGNED
from botocore.config import Config

"""Open Images is a dataset of ~9 million URLs to images that have been
   annotated with labels spanning over 6000 categories.
   This page: https://github.com/cvdfoundation/open-images-dataset
   aims to provide the download instructions
   and mirror sites for Open Images Dataset.
   
   The S3 buckets for images with bounding box are -

    s3://open-images-dataset/train (513GB)
    s3://open-images-dataset/validation (12GB)
    s3://open-images-dataset/test (36GB)
"""

"usage: python data_downloader.py"


class DataDownloader:
    def __init__(self, run_mode, classes):
        self.run_mode = run_mode
        self.classes = classes
    
    def run(self):
        
        self.create_paths()
        
        self.get_meta_files()
        exit()
        self.select_classes_from_meta_file()
        
        self.download_data()
        print("Done!")
    
    def create_dirs(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
    def create_paths(self):
        
        self.DATA_FILES_DIR = "../data/"
        self.JPG_IMAGES_DIR = f"{self.DATA_FILES_DIR}images"
        self.LABELS_DIR = f"{self.DATA_FILES_DIR}labels"
        self.META_FILES_DIR = f"{self.DATA_FILES_DIR}meta_files"
        
        paths_list = [
            self.DATA_FILES_DIR,
            self.JPG_IMAGES_DIR,
            self.LABELS_DIR,
            self.META_FILES_DIR 
            
        ]
        
        for dir_name in paths_list:
            self.create_dirs(dir_name)
            print(dir_name, " created!\n")

    def get_meta_files(self):
        
        # Download required meta-files
        url = "https://storage.googleapis.com/openimages/2018_04/"
        
        links = {"class-descriptions-boxable.csv": f"{url}class-descriptions-boxable.csv",
                #  "train-annotations-bbox.csv": f"{url}train/train-annotations-bbox.csv",
                 "validation-annotations-bbox.csv": f"{url}validation/validation-annotations-bbox.csv",
                 "test-annotations-bbox.csv": f"{url}/test/test-annotations-bbox.csv"
        }
        for file_name, source in links.items():
            
            r = requests.get(source)
            
            with open(f"{self.META_FILES_DIR}/{file_name}", 'w', encoding='utf-8') as file:
                file.write(r.content.decode("utf-8"))
        
    def select_classes_from_meta_file(self):
        
        class_descriptions_file = f"{self.META_FILES_DIR}/class-descriptions-boxable.csv"
        annotations_file = f"{self.META_FILES_DIR}/train-annotations-bbox.csv"
        
        class_descriptions = pd.read_csv(class_descriptions_file, header=None, encoding="utf-8")

        class_descriptions.columns = ["code", "classes"]

        # Find the classes codes
        df_filtered_by_classes = class_descriptions.loc[class_descriptions['classes'].isin(self.classes)]

        code_list = df_filtered_by_classes.code.to_list()

        # Load Annotations File
        annotations = pd.read_csv(annotations_file, encoding="utf-8", low_memory=False)

        # filter annotations by codes
        selected_annotations = annotations.loc[annotations['LabelName'].isin(code_list)]

        selected_annotations = selected_annotations[["ImageID", "XMin", "XMax", "YMin", "YMax"]]
        selected_annotations.index = range(selected_annotations.shape[0])

        del annotations

        print("Total number of annotations : " + str(selected_annotations.shape[0]))

    def download_data(self):
        
        # download the images
        # you can install the aws cli: pip install awscli

        # list the folders: aws s3 --no-sign-request ls s3://open-images-dataset

        # aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/train/000002b66c9c498e.jpg JPEGImages/000002b66c9c498e.jpg

        # here we use the boto3 library
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

        for index, data in selected_annotations.iterrows():
            
            img = data["ImageID"]
            
            xmin = data["XMin"]
            xmax = data["XMax"]
            ymin = data["YMin"]
            ymax = data["YMax"]
            
            file_name = f"{img}.jpg"
            
            with open(f"{self.JPG_IMAGES_DIR}/{file_name}", 'wb') as f:
                s3.download_fileobj('open-images-dataset', f"{self.run_mode}/{file_name}", f)
            
            # Write labels files
            with open(f"{self.LABELS_DIR}/%s.txt"%(img),'a') as f:
                
                f.write(' '.join([str(index), str((float(xmax) + float(xmin))/2),
                                str((float(ymax) + float(ymin))/2),
                                str(float(xmax)-float(xmin)),
                                str(float(ymax)-float(ymin))])+'\n')

if __name__ == "__main__":
    
    data_downloader = DataDownloader(run_mode="train", classes=["Headphones"])
    data_downloader.run()
import gdown
import zipfile
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))

def download_data(url, path_to_save_content, zip_filename):
    zip_path = os.path.join(path_to_save_content, zip_filename)

    # Donwload zip
    gdown.download(url, zip_path, quiet=False, fuzzy=True)

    # Unzip data
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path_to_save_content)

def get_url(dataset_name, content):
    if dataset_name == "birds":
        if content == "data":
            url = "https://drive.google.com/file/d/1CRU0DMCe5UTS12ByCPmC7M3ky1YaLZdt/view?usp=sharing"
        elif content == "encoders":
            url = "https://drive.google.com/file/d/1IlKRU-Migce8Tn6KY-xRs21i7c6Q-58K/view?usp=sharing"
        elif content == "models":
            url = ""
    elif dataset_name == "coco":
        if content == "data":
            url = ""
        elif content == "encoders":
            url = ""
        elif content == "models":
            url = ""
    return url


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="birds")
    parser.add_argument('--content', type=str, default="data", 
                        help="Possible contents to download: [data, encoders, models]")
    args = parser.parse_args()

    url = get_url(args.dataset, args.content)

    path_to_save_content = args.content

    zip_filename = f"{args.dataset}_{args.content}.zip"

    download_data(url, path_to_save_content, zip_filename)
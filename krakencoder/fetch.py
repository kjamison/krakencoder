"""
Functions to retrieve specific pretrained model files, and fetch files from the internet if they are not already present.
Data are stored in the 'krakencoder/model_data' folder in the OS-defined cache location.
If the environment variable KRAKENCODER_DATA is set, use that location instead.
"""

import sys
import argparse
import os
import requests
import hashlib
import json
import pandas as pd
import platformdirs
from tqdm import tqdm


def model_data_folder(data_folder=None, ignore_env=False):
    """
    Returns the folder where the model data is stored 
    By default, the data is stored 'krakencoder/model_data' folder in an OS-defined cache location
    If the environment variable KRAKENCODER_DATA is set, the data is stored in that folder.
    """
    if data_folder is None:
        #scriptdir=os.path.dirname(__file__)
        cachedir=platformdirs.user_cache_dir()
        if 'google.colab' in sys.modules:
            #special mode for google colab
            cachedir='/content/cache'
        data_folder = os.path.join(cachedir, 'krakencoder','model_data')
    if os.environ.get('KRAKENCODER_DATA') and not ignore_env:
        data_folder = os.environ.get('KRAKENCODER_DATA')
    data_folder=os.path.abspath(os.path.expanduser(data_folder))
    return data_folder

def get_fetchable_data_list(override_json=None,filenames_only=False):
    """
    Returns a list of model data files that can be fetched from the internet.
    """
    # Default location for the model data URLs
    # This file should be in the same directory as this script
    # or in the krakencoder package directory
    urlfile=os.path.abspath(os.path.join(os.path.dirname(__file__), 'model_data_urls.json'))
    if override_json is not None:
        # If an override JSON file is provided, use that instead of the default
        urlfile=override_json
    
    if not os.path.exists(urlfile):
        raise FileNotFoundError(f"Model data URLs file not found: {urlfile}. Please provide a valid JSON file with model data URLs.")
    
    try:
        if urlfile.endswith('.json'):
            with open(urlfile,'r') as f:
                data_urls=json.load(f)
        elif urlfile.endswith('.csv'):
            data_urls=pd.read_csv(urlfile).to_dict(orient='records')
        
        if filenames_only:
            data_urls=[data_info['filename'] for data_info in data_urls]
    except:
        data_urls=[]
    
    return data_urls

def fetch_model_data(files_to_fetch=None, data_folder=None, force_download=False, verbose=False):
    """
    Fetches the model data files from the internet. If the files are already present, they are not downloaded again.
    When files are downloaded for the first time, check hash against stored value to ensure integrity.
    
    Parameters:
    files_to_fetch : str or list of str. List of filenames to fetch. If None, all files are fetched.
    data_folder : str (optional). Folder where the data files are stored. If None, the default folder is used.
    force_download : bool (default=False). If True, download the files even if they already exist.
    verbose : bool (default=False). If True, print extra messages about the download status.
        
    Returns:
    data_file_list : str or list of str. List of paths to the downloaded files. If input was a string, this is a string.
    """
    # Fetch data from the internet
    # First find the folder where data should be stored
    data_folder = model_data_folder(data_folder)
    os.makedirs(data_folder, exist_ok=True)
    
    data_urls=get_fetchable_data_list()
    
    input_was_str=False
    
    if files_to_fetch is None:
        # Fetch all files if no input is specified
        files_to_fetch = [data_info['filename'] for data_info in data_urls]
    elif isinstance(files_to_fetch, str):
        input_was_str=True
        files_to_fetch = [files_to_fetch]
        
    data_file_list=[]
    
    for data_file_tofind in files_to_fetch:
        data_info = [data_info for data_info in data_urls if data_info['filename']==data_file_tofind]
        if len(data_info) == 0:
            raise Exception(f"Could not find data file {data_file} in data_urls")
        data_info = data_info[0]
        data_file = data_info['filename']
        url_list = data_info['url']
        if isinstance(url_list,str):
            url_list=[url_list]
        hash_expected = None
        if 'hash' in data_info:
            hash_expected=data_info['hash']
        hash_type = data_info['hashtype']
        
        data_file_path = os.path.join(data_folder, data_file)
        
        check_this_hash = False
        hash_new = None
        if force_download or not os.path.exists(data_file_path):
            # Now we can fetch the data from the internet
            for url in url_list:
                try:
                    print(f"Downloading {data_file_path} from {url}")
                    hash_new=download_url(url, data_file_path, show_progress=True, hash_type=hash_type)
                    check_this_hash=True
                    break
                except Exception as e:
                    print(f" ERROR! Failed to download {data_file_path} from {url}. {e}")
        else:
            if verbose:
                print(f"{data_file_path} already exists. Skipping download.")
        
        if check_this_hash:
            # Check hash only if file was downloaded (not if it already existed)
            if hash_new is None:
                #if we did not compute a hash during download, compute it now
                hash_new=_file_hash(data_file_path, hash_type=hash_type)
            if hash_new == hash_expected:
                if verbose:
                    print(f" SUCCESS! Hash of {data_file_path} matches expected hash")
            else:
                print(f" ERROR! Hash of {data_file_path} does not match expected hash")
                print("  Expected hash: ", hash_expected)
                print("  Computed hash: ", hash_new)
                #if has doesn't match, delete the file and raise an exception
                os.remove(data_file_path)
                raise Exception("Hash mismatch. Please download the file again.")
            
        data_file_list.append(data_file_path)
    
    if input_was_str and len(data_file_list)==1:
        data_file_list=data_file_list[0]
    return data_file_list


def download_url(url, filename, show_progress=False, hash_type=None, hash_expected=None):
    """
    Download a file from the internet with a progress bar.
    Optionally compute the hash of the file as it is downloaded.
    
    Parameters:
    url : str. URL of the file to download.
    filename : str. Path to save the file.
    show_progress : bool (default=False). If True, show a progress bar.
    hash_type : str (default=None). Hash function to use (eg 'sha256'). If None, no hash is computed.
    
    Returns:
    file_hash : str or None. Hash of the downloaded file. None if hash_type is None.
    """
    hasher = None
    if hash_type:
        hasher = hashlib.new(hash_type)
    
    # Make a request to get the content-length
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Set up the tqdm progress bar
    tqdm_bar = None
    if show_progress:
        tqdm_bar = tqdm(total=total_size, unit='B', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(4096):  # Adjust chunk size as necessary
            # Write data to file
            file.write(data)
            
            if hasher is not None:
                hasher.update(data)
            
            if tqdm_bar is not None:
                tqdm_bar.update(len(data))
    
    if tqdm_bar is not None:
        tqdm_bar.close()
    
    if hasher is not None:
        file_hash = hasher.hexdigest()
    else:
        file_hash = None
    
    if hash_expected is not None and file_hash != hash_expected:
        raise ValueError(f"Hash mismatch: expected {hash_expected}, got {file_hash}")
    
    return file_hash

def _file_hash(filename, hash_type='sha256'):
    """
    Compute the hash of a file using a specified hash function.
    """
    hash_func = hashlib.new(hash_type)
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def argument_parse_fetchdata(argv):    
    parser=argparse.ArgumentParser(description=f"""Fetch or clear model data files. 
                                   Data are stored in {model_data_folder(ignore_env=True)}, 
                                   unless environment variable KRAKENCODER_DATA is set.""")
    
    parser.add_argument('--fetch',action='store_true',dest='fetchdata', help='Fetch model data files')
    parser.add_argument('--force',action='store_true',dest='forcedownload', help='Override existing files when fetching data')
    parser.add_argument('--display',action='store_true',dest='displaydata', help='Display model data files')
    
    args=parser.parse_args(argv)
    return args

if __name__ == '__main__':
    if len(sys.argv)<=1:
        argument_parse_fetchdata(['-h'])
        sys.exit(0)
        
    args=argument_parse_fetchdata(sys.argv[1:])
    if args.fetchdata:
        fetch_model_data(force_download=args.forcedownload, verbose=True)
    
    if args.displaydata:
        data_folder = model_data_folder()
        print(f"Contents of data folder: {data_folder}")
        for filename in os.listdir(data_folder):
            print(f"{os.path.join(data_folder,filename)}")
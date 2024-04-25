"""
Functions to retrieve specific pretrained model files, and fetch files from the internet if they are not already present.
"""

import sys
import argparse
import os
import urllib.request
import hashlib


def _file_hash(filename, hash_type='sha256'):
    """
    Compute the hash of a file using a specified hash function.
    """
    hash_func = hashlib.new(hash_type)
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def model_data_folder():
    """
    Returns the folder where the model data is stored (inside package)
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), 'model_data'))

def fetch_model_data(files_to_fetch=None, data_folder=None, verbose=False):
    """
    Fetches the model data files from the internet. If the files are already present, they are not downloaded again.
    When files are downloaded for the first time, check hash against stored value to ensure integrity.
    
    Parameters:
    files_to_fetch : str or list of str. List of filenames to fetch. If None, all files are fetched.
    data_folder : str (optional). Folder where the data files are stored. If None, the default folder is used.
    verbose : bool (default=False). If True, print extra messages about the download status.
        
    Returns:
    data_file_list : str or list of str. List of paths to the downloaded files. If input was a string, this is a string.
    """
    # Fetch data from the internet
    # First find the folder inside this package where data is stored:
    if data_folder is None:
        data_folder = model_data_folder()
    os.makedirs(data_folder, exist_ok=True)
    
    data_urls = [
                {'filename':'kraken_chkpt_SCFC_fs86+shen268+coco439_pc256_225paths_latent128_20240413_ep002000.pt',
                 'url':'https://osf.io/download/x2dq5/',
                 'sha256hash':'680f6e527f8fa8fe692128e28bb82b31936d6a572aee5dded6b57b4b62abefbe'},
                   
                {'filename':'kraken_ioxfm_SCFC_fs86_pc256_710train.npy',
                 'url': 'https://osf.io/download/8jxkm/',
                 'sha256hash':'d8a2e1265539dba96ac0cb1c7405e37ac514f518322ffaeefc5f02c63ea755ca'},
                 
                {'filename':'kraken_ioxfm_SCFC_shen268_pc256_710train.npy',
                 'url': 'https://osf.io/download/z2qpt/',
                 'sha256hash':'f89a527199763a198c4be33a51b913258b4f5538e39b62eeba4700c890d2915e'},
                 
                {'filename':'kraken_ioxfm_SCFC_coco439_pc256_710train.npy',
                 'url': 'https://osf.io/download/tu2mr/',
                 'sha256hash':'afbe4d329f00cd99c0f2af5b4fbb095454744717bd4eb7241132f421f2cbef47'},
                 
                {'filename':'subject_splits_993subj_683train_79val_196test_retestInTest.mat',
                 'url':'https://osf.io/download/y67ep/',
                 'sha256hash':'86fb6be66e2406a4350c6bb9e7221c0f2272287b61c90c76cf27a8e415977a36'}
                ]
    
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
        url = data_info['url']
        hash_expected = None
        if 'sha256hash' in data_info:
            hash_expected=data_info['sha256hash']
        
        data_file_path = os.path.join(data_folder, data_file)
        
        check_this_hash = False
        if not os.path.exists(data_file_path):
            # Now we can fetch the data from the internet
            print(f"Downloading {data_file_path} from {url}")
            urllib.request.urlretrieve(url, data_file_path)
            check_this_hash=True
        else:
            if verbose:
                print(f"{data_file_path} already exists. Skipping download.")
        
        if check_this_hash:
            # Check hash only if file was downloaded (not if it already existed)
            hash_new=_file_hash(data_file_path, hash_type='sha256')
            if hash_new != hash_expected:
                print(f"Hash of {data_file_path} does not match expected hash")
                print(" Expected hash: ", hash_expected)
                print(" Computed hash: ", hash_new)
                #if has doesn't match, delete the file and raise an exception
                os.remove(data_file_path)
                raise Exception("Hash mismatch. Please download the file again.")
            
        data_file_list.append(data_file_path)
    
    if input_was_str and len(data_file_list)==1:
        data_file_list=data_file_list[0]
    return data_file_list

def clear_model_data(data_folder=None, verbose=True):
    """
    Delete all model data files from the specified folder.
    
    Parameters:
    data_folder : str (optional). Folder where the data files are stored. If None, the default folder is used.
    """
    if data_folder is None:
        data_folder = model_data_folder()
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                if verbose:
                    print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def argument_parse_fetchdata(argv):    
    parser=argparse.ArgumentParser(description='Fetch or clear model data files')
    
    parser.add_argument('--clear',action='store_true',dest='cleardata', help='Clear model data files')
    parser.add_argument('--fetch',action='store_true',dest='fetchdata', help='Fetch model data files')
    parser.add_argument('--display',action='store_true',dest='displaydata', help='Display model data files')
    
    args=parser.parse_args(argv)
    return args

if __name__ == '__main__':
    if len(sys.argv)==1:
        argument_parse_fetchdata(['-h'])
        sys.exit(0)
        
    args=argument_parse_fetchdata(sys.argv[1:])
    if args.cleardata:
        clear_model_data(verbose=True)
    elif args.fetchdata:
        fetch_model_data(verbose=True)
    
    if args.displaydata:
        data_folder = model_data_folder()
        print(f"Contents of data folder: {data_folder}")
        for filename in os.listdir(data_folder):
            print(f"{os.path.join(data_folder,filename)}")
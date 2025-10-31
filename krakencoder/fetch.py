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


try:
    from ._resources import resource_path
except ImportError:
   #in case it is being called directly from command line
   from _resources import resource_path

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


def load_flavor_database(dbfile=None, conntype_list=None, directory_search_list=[], override_abs_path=False, 
                           fields_to_check=['checkpoint','xform'],
                           only_return_exists_or_fetchable=True):
    """
    Load a json file with flavor information, including the following fields:
    flavorinfo[conntype]['atlas']: str, atlas name
    flavorinfo[conntype]['checkpoint']: str, "*.pt" checkpoint filename trained on this flavor (could include multiple flavors)
    flavorinfo[conntype]['xform']: str, "*.npy" filename for the pre-computed transformer for this flavor (Eg: PCA)
    flavorinfo[conntype]['data']: str, filename of the input data for this flavor
    
    flavorinfo[conntype]['<field>_exists']: bool, True if a given file has been found (absolute path)
    flavorinfo[conntype]['<field>_fetchable']: bool, True if a given file can be fetched from the uploaded database
    
    Filenames stored in json can be relative names without paths. This function will search for the files in the directory_search_list
     
    If override_abs_path=True, even if input json had an absolute path, search for the filename in the directory list anyway

    Return a dict with only the flavors specified in the conntype_list (or all flavors if conntype_list is None)
    """
    if dbfile is None:
        #default to the model_data_urls.json file in the same directory as this script
        dbfile=os.path.abspath(resource_path('flavordb.json'))
    
    if isinstance(conntype_list,str):
        conntype_list=[conntype_list]
    if isinstance(directory_search_list,str):
        directory_search_list=[directory_search_list]
    
    if len(directory_search_list)==0:
        directory_search_list=[model_data_folder()]
        
    if dbfile.endswith('.json'):
        with open(dbfile,'r') as f:
            flavor_input_info=json.load(f)
    elif dbfile.endswith('.csv'):
        flavor_input_info=pd.read_csv(dbfile).set_index('flavor').to_dict(orient='index')
    elif dbfile.endswith('.tsv'):
        flavor_input_info=pd.read_csv(dbfile,sep='\t').set_index('flavor').to_dict(orient='index')
    else:
        raise Exception(f"Unsupported file format for {dbfile}. Please provide a JSON, CSV, or TSV file.")
    
    if conntype_list is not None:
        flavor_input_info={k:flavor_input_info[k] for k in conntype_list}
    
    fetchable_urls=get_fetchable_data_list()
    fetchable_files=[u['filename'] for u in fetchable_urls]
    
    #filename_fields=[f for f in fields_to_check if f in ['checkpoint','xform','data']]
    if fields_to_check=='all' or fields_to_check==['all']:
        k=list(flavor_input_info.keys())[0]
        filename_fields=list(flavor_input_info[k].keys())
    else:
        filename_fields=[f for f in fields_to_check]
    
    for k in flavor_input_info:
        all_exist=True
        all_fetchable=True
        all_exists_or_fetchable=True
        for f in filename_fields:
            if f not in flavor_input_info[k]:
                #if the field is not in the flavor info, skip it
                raise Exception(f"Field '{f}' not found in flavor info for {k}. Available fields: {list(flavor_input_info[k].keys())}")
            is_fetchable=flavor_input_info[k][f] in fetchable_files
            found_absolute=False
            if os.path.exists(flavor_input_info[k][f]):
                found_absolute=True
            elif os.path.exists(os.path.abspath(os.path.expanduser(flavor_input_info[k][f]))):
                flavor_input_info[k][f]=os.path.abspath(os.path.expanduser(flavor_input_info[k][f]))
                found_absolute=True
            else:
                fname=flavor_input_info[k][f]
                for d in directory_search_list:
                    d=os.path.abspath(os.path.expanduser(d))
                    if os.path.exists(os.path.join(d,fname)):
                        found_absolute=True
                        flavor_input_info[k][f]=os.path.join(d,fname)
                        break
                    elif override_abs_path and os.path.exists(os.path.join(d, os.path.split(fname)[-1])):
                        found_absolute=True
                        flavor_input_info[k][f]=os.path.join(d,os.path.split(fname)[-1])
                        break
            flavor_input_info[k][f'{f}_exists']=found_absolute
            flavor_input_info[k][f'{f}_fetchable']=is_fetchable
            all_exist=all_exist and found_absolute
            all_fetchable=all_fetchable and is_fetchable
            all_exists_or_fetchable=all_exists_or_fetchable and (found_absolute or is_fetchable)
        flavor_input_info[k]['all_exists']=all_exist
        flavor_input_info[k]['all_fetchable']=all_fetchable
        flavor_input_info[k]['all_exists_or_fetchable']=all_exists_or_fetchable
    
    if only_return_exists_or_fetchable:
        #filter out flavors that are not fetchable or do not have all files
        flavor_input_info={k:v for k,v in flavor_input_info.items() if v['all_exists_or_fetchable']}
    
    return flavor_input_info

def get_fetchable_data_list(override_json=None,filenames_only=False):
    """
    Returns a list of model data files that can be fetched from the internet.
    """
    # Default location for the model data URLs
    # This file should be in the same directory as this script
    # or in the krakencoder package directory
    urlfile=os.path.abspath(resource_path('model_data_urls.json'))
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
        elif urlfile.endswith('.tsv'):
            data_urls=pd.read_csv(urlfile, sep='\t').to_dict(orient='records')
        else:
            raise Exception(f"Unsupported file format for {urlfile}. Please provide a JSON, CSV, or TSV file.")
        
        if filenames_only:
            data_urls=[data_info['filename'] for data_info in data_urls]
    except:
        data_urls=[]
    
    return data_urls

def replace_data_folder_placeholder(filename, data_folder=None):
    """
    Replace placeholder strings in the URLs with the actual data folder path.
    """
    
    if isinstance(filename,list):
        return [replace_data_folder_placeholder(f, data_folder=data_folder) for f in filename]
    
    if filename is None:
        return filename
    
    for d in ['{FETCH}','{KRAKENCODER_DATA}','{KRAKENDATA}','{KRAKENDATAFOLDER}']:
        if d in filename:
            filename=filename.replace(d,model_data_folder(data_folder))
    
    return filename

def is_fetchable_file(filename, data_folder=None):
    """
    Check if a given filename is in the list of fetchable data files.
    """
    fetchable_list=get_fetchable_data_list(filenames_only=True)
    filename=replace_data_folder_placeholder(filename, data_folder=data_folder)
    if filename.startswith(model_data_folder(data_folder)):
        filename=os.path.basename(filename)
    return filename in fetchable_list

def fetch_model_data_if_needed(files_to_fetch=None, data_folder=None, force_download=False, verbose=False):
    """
    Fetch model data files if they are fetchable and not already present, otherwise returns input path(s)
    
    Parameters:
    files_to_fetch : str or list of str. List of filenames to fetch. if None, do nothing. Can be NAME=FILE to preserve prefix.
    data_folder : str (optional). Folder where the data files are stored. If None, the default folder is used.
    force_download : bool (default=False). If True, download the files even if they already exist.
    verbose : bool (default=False). If True, print extra messages about the download status.
    
    Returns:
    data_file_list : str or list of str. List of paths to the downloaded files OR input files. If input was a string, this is a string.
    """
    
    if files_to_fetch is None:
        return None
    
    if isinstance(files_to_fetch,str):
        input_was_str=True
        files_to_fetch=[files_to_fetch]
    else:
        input_was_str=False
    
    files_new=[]
    for f in files_to_fetch:
        f=replace_data_folder_placeholder(f)
        fstart=None
        if '=' in f:
            fstart, f = f.split('=',1)
        if f.startswith(model_data_folder(data_folder)):
            f=os.path.basename(f)
        if is_fetchable_file(f, data_folder=data_folder):
            f=fetch_model_data(files_to_fetch=f, data_folder=data_folder, force_download=force_download, verbose=verbose)
        if fstart is not None:
            f=fstart+'='+f
        files_new+=[f]
    
    if input_was_str:
        return files_new[0]
    else:
        return files_new

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
    
    files_to_fetch=[replace_data_folder_placeholder(f) for f in files_to_fetch]
    
    data_file_list=[]
    
    for data_file_tofind in files_to_fetch:
        if data_file_tofind.startswith(model_data_folder()):
            data_file_tofind=os.path.basename(data_file_tofind)
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
                    
                    check_this_hash=False
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
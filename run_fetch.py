#!/usr/bin/env python3

"""
Command-line script to fetch models files for applying Krakencoder to a dataset
"""

import os
import sys
import argparse

from krakencoder.fetch import model_data_folder, get_fetchable_data_list, fetch_model_data, load_flavor_database

def argument_parse_fetchscript(argv):
    parser=argparse.ArgumentParser(description=f"""Fetch or clear model data files. 
                                   Data are stored in {model_data_folder(ignore_env=True)}, 
                                   unless environment variable KRAKENCODER_DATA is set.""")
    
    parser.add_argument('--fetchall',action='store_true',dest='fetchalldata', help='Fetch ALL model data files')
    parser.add_argument('--force',action='store_true',dest='forcedownload', help='Override existing files when fetching data')
    parser.add_argument('--display',action='store_true',dest='displaydata', help='Display model data files')
    parser.add_argument('--listfetchablefiles',action='store_true',dest='listfetchablefiles', help='Display list of fetchable model data files')
    parser.add_argument('--listflavors',action='store_true',dest='listflavors', help='Display list of fetchable flavors')
    parser.add_argument('--listflavorfiles',action='store_true',dest='listflavorfiles', help='Display list of files for fetchable flavors')
    parser.add_argument('--printflavorxform',action='store',dest='printxform', help='Print name=file xform filename for each flavor',nargs='*')
    parser.add_argument('--printflavorcheckpoint',action='store',dest='printcheckpoint', help='Print name=file checkpoint filename for each flavor',nargs='*')
    parser.add_argument('--uniqueflavorcheckpoint',action='store',dest='printuniquecheckpoint', help='Print unique checkpoint filenames for a given list of flavors',nargs='*')
    parser.add_argument('--uniqueflavorxform',action='store',dest='printuniquexform', help='Print unique xform filenames for a given list of flavors',nargs='*')
    parser.add_argument('--fetch',action='store',dest='fetchfile', help='Fetch model data file by filename', nargs='*')
    parser.add_argument('--fetchflavor','--fetchflavors',action='store',dest='fetchflavor', help='Fetch model data file by flavor name', nargs='*')
    parser.add_argument('--fetchtypes',action='store',dest='fetchtypes', help='Which file types to download (eg: "checkpoint","xform")', nargs='*')
    
    return parser.parse_args(argv)


def run_fetchdata(argv=None):
    if argv is None:
        argv=sys.argv[1:]
    #read in command-line inputs
    args=argument_parse_fetchscript(argv)
    
    flavor_info=load_flavor_database()
    
    data_folder = model_data_folder()
    data_files_downloaded = os.listdir(data_folder) if os.path.exists(data_folder) else []
    
    if args.fetchtypes is None or len(args.fetchtypes)==0:
        args.fetchtypes=['checkpoint','xform']
    for i,t in enumerate(args.fetchtypes):
        if t in ['xform','ioxfm']:
            t='xform'
        elif t in ['checkpoint','model','pt']:
            t='checkpoint'
        else:
            print(f"Unknown file type {t}. Supported types are 'checkpoint' and 'xform'.")
            sys.exit(1)
        args.fetchtypes[i]=t
    
    if args.printxform is not None and len(args.printxform)>0:
        xformlist=[]
        for c in args.printxform:
            if c not in flavor_info:
                continue
            if 'xform' in flavor_info[c] and flavor_info[c]['xform'] is not None:
                ioxfm=flavor_info[c]['xform']
                ioxfm=os.path.basename(ioxfm)
                xformlist+=[f"{c}={ioxfm}"]
        print(" ".join(xformlist))
        sys.exit(0)
    
    if args.printcheckpoint is not None and len(args.printcheckpoint)>0:
        ptlist=[]
        for c in args.printcheckpoint:
            if c not in flavor_info:
                continue
            if 'checkpoint' in flavor_info[c] and flavor_info[c]['checkpoint'] is not None:
                ptfile=flavor_info[c]['checkpoint']
                ptfile=os.path.basename(ptfile)
                ptlist+=[f"{c}={ptfile}"]
        print(" ".join(ptlist))
        sys.exit(0)
    
        
    if args.printuniquecheckpoint is not None and len(args.printuniquecheckpoint)>0:
        ptlist=[]
        for c in args.printuniquecheckpoint:
            if c not in flavor_info:
                continue
            if 'checkpoint' in flavor_info[c] and flavor_info[c]['checkpoint'] is not None:
                ptfile=flavor_info[c]['checkpoint']
                ptfile=os.path.basename(ptfile)
                ptlist+=[ptfile]
        ptlist=list(set(ptlist))
        print(" ".join(ptlist))
        sys.exit(0)
    
    if args.printuniquexform is not None and len(args.printuniquexform)>0:
        xformlist=[]
        for c in args.printuniquexform:
            if c not in flavor_info:
                continue
            if 'xform' in flavor_info[c] and flavor_info[c]['xform'] is not None:
                ioxfm=flavor_info[c]['xform']
                ioxfm=os.path.basename(ioxfm)
                xformlist+=[ioxfm]
        xformlist=list(set(xformlist))
        print(" ".join(xformlist))
        sys.exit(0)
    
    if args.listflavors:
        conntypes=[c for c in flavor_info if flavor_info[c]['all_exists_or_fetchable']]
        print(" ".join(conntypes))
        sys.exit(0)
    
    if args.listflavorfiles:
        print("Available flavors:")
        for c in flavor_info:
            status=""
            if not flavor_info[c]['all_exists_or_fetchable']:
                status="not fetchable"
            if flavor_info[c]['all_exists']:
                status="all files present"
            elif flavor_info[c]['all_fetchable']:
                status="all files fetchable"
            print(f"{c}: {status}")
            for t in args.fetchtypes:
                if t in flavor_info[c] and flavor_info[c][t] is not None:
                    fname=flavor_info[c][t]
                    fname=os.path.basename(fname)
                    status=u'\u2713' if fname in data_files_downloaded else " "
                    print(f"  [{status}] {t}: {fname}")
                else:
                    print(f"  [ ] {t}: [none]")
        sys.exit(0)
    
    if args.listfetchablefiles:
        fetchable_list=get_fetchable_data_list()
        print("Fetchable model data files:")
        for item in fetchable_list:
            fname=item['filename']
            if fname in data_files_downloaded:
                status=u'\u2713'
            else:
                status=" "
            print(f"[{status}] {item['filename']} ")
        sys.exit(0)
    
    if args.fetchfile is not None and len(args.fetchfile)>0:
        fetchable_list=get_fetchable_data_list()
        fetchable_filenames=[item['filename'] for item in fetchable_list]
        files_to_fetch=[]
        for fname in args.fetchfile:
            if fname not in fetchable_filenames:
                print(f"File {fname} is not in the list of fetchable files. Use --listfetchablefiles to see available files.")
                continue
            files_to_fetch+=[fname]
        try:
            if len(files_to_fetch)>0:
                fetch_model_data(files_to_fetch=files_to_fetch, force_download=args.forcedownload, verbose=True)
        except Exception as e:
            print(f"Error fetching files {files_to_fetch}: {e}")
    
    if args.fetchflavor is not None and len(args.fetchflavor)>0:
        fetchable_list=get_fetchable_data_list()
        fetchable_filenames=[item['filename'] for item in fetchable_list]
        files_to_fetch=[]
        for f in args.fetchflavor:
            if not f in flavor_info:
                print(f"Flavor {f} not found in flavor database.")
                continue
            for t in args.fetchtypes:
                if not t in flavor_info[f] or flavor_info[f][t] is None:
                    print(f"Flavor {f} does not have a file of type {t}.")
                    continue
                fname=flavor_info[f][t]
                fname=os.path.basename(fname)
                files_to_fetch+=[fname]
        try:
            if len(files_to_fetch)>0:
                fetch_model_data(files_to_fetch=files_to_fetch, force_download=args.forcedownload, verbose=True)
        except Exception as e:
            print(f"Error fetching files {files_to_fetch}: {e}")
    
    if args.fetchalldata:
        fetch_model_data(force_download=args.forcedownload, verbose=True)
    
    if args.displaydata:
        data_folder = model_data_folder()
        print(f"Contents of data folder: {data_folder}")
        for filename in os.listdir(data_folder):
            print(f"{os.path.join(data_folder,filename)}")

if __name__ == "__main__":
    if len(sys.argv)<=1:
        argument_parse_fetchscript(['-h'])
        sys.exit(0)
    run_fetchdata(sys.argv[1:])

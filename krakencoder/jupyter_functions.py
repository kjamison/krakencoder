"""
Functions for Jupyter notebooks
"""

import os
import sys
import io

import zipfile
import numpy as np
from scipy.io import loadmat, savemat
import importlib.util

import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import json
import re
from tqdm.auto import tqdm

try:
    colab_spec = importlib.util.find_spec("google.colab")
    from google.colab import drive as colab_drive
    from google.colab import files as colab_files
except ImportError:
    colab_spec = None


class SuppressOutput:
    """
    Function to suppress the output (ie print statements when mounting google drive)
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


def jupyter_create_upload_widget(
    loadfile_callback=None,
    extensions=None,
    multiple=False,
    initial_gdrive_path="/content/drive",
    initial_local_path=".",
    initial_tab="upload",
):
    # Step 4: Function to list directories and files in a directory

    folder_display_prefix = "\U0001F4C1 "
    file_display_prefix = "\U0001F4C4 "

    is_colab = colab_spec is not None

    filetab_titles = {
        "upload": "Upload",
        "local": "Browse Files",
        "gdrive": "Google Drive",
    }
    if is_colab:
        filetab_list = ["upload", "local", "gdrive"]
    else:
        filetab_list = ["upload", "local"]

    def clean_filename_decoration(filename):
        if filename.startswith(folder_display_prefix):
            filename = filename[len(folder_display_prefix) :]
        if filename.startswith(file_display_prefix):
            filename = filename[len(file_display_prefix) :]
        if filename.startswith(". ["):
            filename = "."
        elif filename.startswith(".. ["):
            filename = ".."
        return filename

    def list_files_and_dirs(path):

        items = os.scandir(path)
        items = [
            (
                folder_display_prefix + f.name + "/"
                if f.is_dir()
                else file_display_prefix + f.name
            )
            for f in items
        ]
        items = [f for f in items if any(f.endswith(ext) for ext in ["/"] + extensions)]
        items.sort()

        # directories first, then files
        items = [f for f in items if f.endswith("/")] + [
            f for f in items if not f.endswith("/")
        ]

        # items=['.. ['+os.path.dirname(path)+']']+items
        # items=['. ['+os.path.basename(path)+']']+items
        # items=['.. [Parent Dir]']+items
        # items=['. []']+items

        # add self and parent directories
        items = [folder_display_prefix + ".", folder_display_prefix + ".."] + items

        return items

    # widget for printing output during callback functions
    loadfile_output_widget = widgets.Output()

    #################
    # LOCAL FILE TAB

    local_item_selector_label = widgets.Label(value="Browsing local files...")

    # Step 5: Create dropdown widget for file and directory selection
    local_item_selector = widgets.Select(
        options=[""], disabled=False, rows=10, layout=widgets.Layout(width="50%")
    )
    local_item_selector.current_path = initial_local_path

    # Step 7: Button to trigger data processing
    local_process_button = widgets.Button(
        description="Add File",
        disabled=False,
        button_style="",
        tooltip="Click to process the selected file",
        icon="check",
    )
    # Step 7: Button to trigger data processing
    local_refresh_button = widgets.Button(
        description="Refresh",
        disabled=False,
        button_style="",
        tooltip="Click to refresh the file list",
        icon="refresh",
    )
    
    # Step 8: Define the item selection event handler
    def local_on_item_selector_double_click(change):
        current_path = change["owner"].current_path  #!test
        selected_item = change["new"]

        if selected_item is None:
            local_process_button.disabled = False
            return

        # clean up some file text decoration
        selected_item = clean_filename_decoration(selected_item)

        selected_path = os.path.abspath(os.path.join(current_path, selected_item))
        if selected_path == current_path:
            return

        if os.path.isdir(selected_path):
            # If it's a directory, update the list to show its contents
            current_path = selected_path
            local_item_selector.current_path = current_path  #!test
            local_update_item_selector(current_path)
        else:
            # If it's a file, enable the process button
            local_process_button.disabled = False

    # Step 9: Define the button click event handler for file processing
    def local_on_process_button_clicked(b):
        current_path = local_item_selector.current_path
        selected_file = clean_filename_decoration(local_item_selector.value)
        file_path = os.path.join(current_path, selected_file)
        if loadfile_callback is not None:
            with loadfile_output_widget:
                loadfile_callback(file_path)
    
    local_refresh_button.on_click(lambda b: local_update_item_selector(local_item_selector.current_path))

    local_process_button.on_click(local_on_process_button_clicked)

    # Step 10: Function to update the item selector
    def local_update_item_selector(path):
        items = list_files_and_dirs(path)
        local_item_selector.options = items
        local_item_selector_label.value = "Browsing: " + os.path.abspath(path)
        local_process_button.disabled = True

    #################
    # GOOGLE DRIVE TAB
    gdrive_item_selector_label = widgets.Label(
        value="Requesting access to Google Drive..."
    )

    # Step 5: Create dropdown widget for file and directory selection
    gdrive_item_selector = widgets.Select(
        options=[""], disabled=False, rows=10, layout=widgets.Layout(width="50%")
    )
    gdrive_item_selector.current_path = initial_gdrive_path

    # Step 6: Define a data processing function
    def data_processing_function(file_path):
        # Example: Read a CSV file and display the first 5 rows
        data = pd.read_csv(file_path)
        display(data.head())

    # Step 7: Button to trigger data processing
    gdrive_process_button = widgets.Button(
        description="Add File",
        disabled=False,
        button_style="",
        tooltip="Click to process the selected file",
        icon="check",
    )
    
    gdrive_refresh_button = widgets.Button(
        description="Refresh",
        disabled=False,
        button_style="",
        tooltip="Click to refresh the file list",
        icon="refresh",
    )
    # Step 8: Define the item selection event handler
    def gdrive_on_item_selector_double_click(change):
        current_path = change["owner"].current_path  #!test
        selected_item = change["new"]

        if selected_item is None:
            gdrive_process_button.disabled = False
            return

        # clean up some file text decoration
        selected_item = clean_filename_decoration(selected_item)

        selected_path = os.path.abspath(os.path.join(current_path, selected_item))
        if selected_path == current_path:
            return

        if os.path.isdir(selected_path):
            # If it's a directory, update the list to show its contents
            current_path = selected_path
            gdrive_item_selector.current_path = current_path  #!test
            gdrive_update_item_selector(current_path)
        else:
            # If it's a file, enable the process button
            gdrive_process_button.disabled = False

    # Step 9: Define the button click event handler for file processing
    def gdrive_on_process_button_clicked(b):
        current_path = gdrive_item_selector.current_path
        selected_file = clean_filename_decoration(gdrive_item_selector.value)
        file_path = os.path.join(current_path, selected_file)
        if loadfile_callback is not None:
            with loadfile_output_widget:
                loadfile_callback(file_path)
    
    gdrive_process_button.on_click(gdrive_on_process_button_clicked)
    
    # Step 10: Function to update the item selector
    def gdrive_update_item_selector(path):
        items = list_files_and_dirs(path)
        gdrive_item_selector.options = items
        gdrive_item_selector_label.value = "Browsing: " + os.path.abspath(path)
        gdrive_process_button.disabled = True
    
    gdrive_refresh_button.on_click(lambda b: gdrive_update_item_selector(gdrive_item_selector.current_path))
    
    ###############
    # FILE UPLOADER TAB
    def on_file_upload(change):
        uploaded_file_dict = change["new"]

        # non-colab jupyter notebooks seem to return the uploaded file in a different form
        if type(uploaded_file_dict) is tuple:
            uploaded_file_dict = uploaded_file_dict[0]
        if "name" in uploaded_file_dict:
            uploaded_file_dict = {uploaded_file_dict["name"]: uploaded_file_dict}

        # do_write_file=True: write uploaded zip to notebook storage, and read as normal file
        # do_write_file=False: just pass contents as data buffer to callback

        # if we're in colab, we can write the file to the local drive, which makes it easier to
        # work with the file in the future. Otherwise, just load in the file as a buffer.
        do_write_file = is_colab

        for filename, filedata in uploaded_file_dict.items():
            if do_write_file:
                with open(filename, "wb") as f:
                    f.write(filedata["content"])
            if loadfile_callback is not None:
                with loadfile_output_widget:
                    if do_write_file:
                        loadfile_callback(filename, allowed_extensions=extensions)
                    else:
                        loadfile_callback(
                            filename,
                            filebytes=io.BytesIO(filedata["content"]),
                            allowed_extensions=extensions,
                        )

    ###################

    def on_filetab_change(change):
        current_path = gdrive_item_selector.current_path
        tabtype = filetab_list[change["new"]]
        tabtitle = filetab_titles[tabtype]
        if tabtype == "gdrive":
            if not is_colab:
                gdrive_item_selector_label.value = "Google Drive is not available"
                return
            try:
                if not os.path.exists("/content/drive"):
                    with SuppressOutput():
                        colab_drive.mount("/content/drive", readonly=True)
                gdrive_update_item_selector(current_path)
            except:
                pass
        elif tabtype == "local":
            local_update_item_selector(initial_local_path)

    # Step 11: Display the widgets

    title_list = []
    tab_list = []
    for i, t in enumerate(filetab_list):
        title_list.append(filetab_titles[t])
        if t == "gdrive":
            drivebox = widgets.VBox(
                [
                    gdrive_item_selector_label,
                    gdrive_item_selector,
                    widgets.HBox([gdrive_process_button, gdrive_refresh_button])
                ]
            )
            gdrive_item_selector.observe(
                gdrive_on_item_selector_double_click, names="value"
            )
            tab_list.append(drivebox)
        elif t == "local":
            localbox = widgets.VBox(
                [local_item_selector_label, 
                 local_item_selector, 
                 widgets.HBox([local_process_button,local_refresh_button])]
            )
            local_item_selector.observe(
                local_on_item_selector_double_click, names="value"
            )
            tab_list.append(localbox)
        elif t == "upload":
            uploadbox = widgets.FileUpload(
                accept=",".join(extensions),
                multiple=multiple,
                description="Upload file",
            )
            uploadbox.observe(on_file_upload, names="value")
            tab_list.append(uploadbox)

    tabs = widgets.Tab(titles=title_list, children=tab_list)
    for i, s in enumerate(title_list):
        tabs.set_title(i, s)

    tabs.observe(on_filetab_change, names="selected_index")

    # set initial tab to the second tab in the list
    initial_tab_index = [i for i, t in enumerate(filetab_list) if t == initial_tab]
    if len(initial_tab_index) > 0:
        tabs.selected_index = initial_tab_index[0]
    display(tabs, loadfile_output_widget)


def jupyter_create_save_widget(
    savefile_callback=None,
    outvariable_default=None,
    outfile_default="exampledata_outputs.mat",
):
    is_colab = colab_spec is not None

    # widget for printing output during callback functions
    savefile_output_widget = widgets.Output()

    outputvariable_textbox_label = widgets.Label(value="Save variable ")
    outputvariable_textbox = widgets.Text(
        value=outvariable_default, layout=widgets.Layout(width="15em")
    )
    outputfile_textbox_label = widgets.Label(value=" to file ")
    outputfile_textbox = widgets.Text(value=outfile_default)
    outputfile_savebutton = widgets.Button(description="Save")
    outputfile_downloadbutton = widgets.Button(description="Click to download")

    def on_save_clicked(b):
        is_colab = colab_spec is not None
        outfile = outputfile_textbox.value
        outvar = outputvariable_textbox.value
        if savefile_callback is not None:
            with savefile_output_widget:
                savefile_callback(variablename=outvar, filename=outfile)

    def on_download_clicked(b):
        is_colab = colab_spec is not None
        if not is_colab:
            return
        outfile = outputfile_textbox.value
        outvar = outputvariable_textbox.value
        if not os.path.exists(outfile) and savefile_callback is not None:
            with savefile_output_widget:
                savefile_callback(variablename=outvar, filename=outfile)
        colab_files.download(outfile)

    outputfile_savebutton.on_click(on_save_clicked)
    hboxlist = [
        outputvariable_textbox_label,
        outputvariable_textbox,
        outputfile_textbox_label,
        outputfile_textbox,
        outputfile_savebutton,
    ]

    if is_colab:
        outputfile_downloadbutton.on_click(on_download_clicked)
        hboxlist.append(outputfile_downloadbutton)

    display(widgets.HBox(hboxlist), savefile_output_widget)


def save_data_zip(
    filename, conndata_squaremats, participants_info, bids_desc=None, verbose=False, filetype='tsv'
):
    desc_str = ""
    if bids_desc is not None:
        desc_str = "_desc-%s" % (bids_desc)
    zipargs = {"compression": zipfile.ZIP_DEFLATED, "compresslevel": 6}
    filecount=0
    totalfilecount=len(conndata_squaremats) * len(participants_info)
    pbar=tqdm(total=totalfilecount, desc="Saving data to zip")
    with zipfile.ZipFile(filename, "w", **zipargs) as zip_ref:
        outfile = io.BytesIO()
        participants_info.to_csv(outfile, sep="\t", index=None)
        outfile.seek(0)
        zip_ref.writestr("participants.tsv", outfile.getvalue())
        for conntype in conndata_squaremats:
            for i, conndata in enumerate(conndata_squaremats[conntype]):
                outfile = io.BytesIO()
                if filetype.lower() == 'mat':
                    savemat(outfile, {'data': conndata.astype(np.float32)}, format="5", do_compression=True)
                    conn_ext='.mat'
                elif filetype.lower() == 'tsv':
                    np.savetxt(outfile, conndata, fmt="%.6f", delimiter="\t")
                    conn_ext='.tsv'
                else:
                    raise ValueError("Unsupported file type: %s" % (filetype))
                outfile.seek(0)
                subjid = participants_info["participant_id"][i]
                if subjid.startswith("sub-"):
                    subjid = subjid[4:]
                conndata_filename = "sub-%s_%s%s_relmat.dense%s" % (
                    subjid,
                    flavor_to_bids_string(conntype),
                    desc_str,
                    conn_ext
                )
                filecount+=1
                #if verbose:
                #    tqdm.write("Adding %s" % (conndata_filename))
                pbar.set_description("Adding %s" % (conndata_filename))
                pbar.update(1)
                zip_ref.writestr(conndata_filename, outfile.getvalue())
    pbar.close()

def flavor_to_bids_string(flavor):
    atlasname = ""
    participant_id = ""
    subject = ""
    flavor_prefix = ""
    flavor_suffix = ""
    
    meas_str = ""
    desc_str = ""

    if "fccorr" in flavor.lower():
        meas_str += "FCcorr"
    elif "fcpcorr" in flavor.lower():
        meas_str += "FCpcorr"
    elif "ifod2act" in flavor.lower():
        meas_str = "SCifod2act"
    elif "sdstream" in flavor.lower():
        meas_str = "SCsdstream"
    else:
        raise Exception("Unknown meas: %s" % (flavor))

    if "hpf" in flavor.lower():
        meas_str += "HPF"
    elif "bpf" in flavor.lower():
        meas_str += "BPF"
    elif "nofilt" in flavor.lower():
        meas_str += "NF"

    if "gsr" in flavor.lower():
        meas_str += "GSR"

    if "sift2count" in flavor.lower() or "sift2_count" in flavor.lower() or flavor.lower().endswith("sift2"):
        meas_str += "SiftCount"
    elif "sift2volnorm" in flavor.lower() or "sift2_volnorm" in flavor.lower():
        meas_str += "SiftVN"
    elif "volnorm" in flavor.lower():
        meas_str += "VN"
    elif "count" in flavor.lower():
        meas_str += "Count"

    atlasname=flavor.split("_")[1].lower()
    
    bids_str = "atlas-%s_meas-%s" % (atlasname, meas_str)
    if desc_str:
        bids_str += "_desc-%s" % (desc_str)

    return bids_str


def parse_bids_string(bids_str):
    bids_str = os.path.basename(bids_str)
    bids_str = bids_str.split(".")[0]
    bids_parts = bids_str.split("_")

    atlasname = ""
    participant_id = ""
    subject = ""
    flavor_prefix = ""
    flavor_suffix = ""

    for s in bids_parts:
        if "-" in s:
            s_name = s.split("-")[0].lower()
            s_val = "-".join(s.split("-")[1:])
            s_val_lower = s_val.lower()
            if s_name == "sub":
                participant_id = s
                subject = s_val

            elif s_name == "atlas" or s_name.lower() == "seg":
                atlasname = s_val.lower()

            elif s_name == "meas":
                if any([s_val_lower.startswith(p) for p in ["fccorr", "fccov"]]):
                    flavor_prefix = "FCcorr_"
                elif s_val_lower.startswith("fcpcorr"):
                    flavor_prefix = "FCpcorr_"
                elif s_val_lower.startswith("scifod2act"):
                    flavor_prefix = "SCifod2act_"
                elif s_val_lower.startswith("scsdstream"):
                    flavor_prefix = "SCsdstream_"
                else:
                    raise Exception("Unknown meas-: %s" % (s_val))

                if flavor_prefix.startswith("FC"):
                    if "hpf" in s_val_lower:
                        flavor_suffix = "_hpf"
                    elif "bpf" in s_val_lower:
                        flavor_suffix = "_bpf"
                    elif s_val_lower.endswith("nf") or s_val_lower.endswith("nfgsr"):
                        flavor_suffix = "_nofilt"
                    if "gsr" in s_val_lower:
                        flavor_suffix += "gsr"
                elif flavor_prefix.startswith("SC"):
                    if "siftvn" in s_val_lower:
                        flavor_suffix = "_sift2volnorm"
                    elif "vn" in s_val_lower:
                        flavor_suffix = "_volnorm"
                    elif "siftcount" in s_val_lower:
                        flavor_suffix = "_sift2"
                    elif "count" in s_val_lower:
                        flavor_suffix = "_count"
                        

    flavorname = flavor_prefix + atlasname + flavor_suffix

    return {
        "participant_id": participant_id,
        "subject": subject,
        "inputtype": flavorname,
    }


def load_data_zip(filename, filebytes=None, allowed_extensions=None):
    # output: conndata_squaremats['conntype']=list([roi x roi])
    #         participants_info
    conndata_squaremats = {}
    conndata_participants = {}
    participants_info = None

    try:
        #first try to load it as a nemoSC zip
        conndata_squaremats, participants_info = load_nemodata_zip(filename, filebytes=filebytes, bidsify_subjects=True)
        return conndata_squaremats, participants_info
    except:
        pass
    
    if filebytes is not None:
        filename_or_filebytes = filebytes
    else:
        filename_or_filebytes = filename

    if allowed_extensions is None:
        allowed_extensions_in_zip = None
    else:
        allowed_extensions_in_zip = set(allowed_extensions) - set(["zip"])
        if len(allowed_extensions_in_zip) == 0:
            allowed_extensions_in_zip = None
    
    with zipfile.ZipFile(filename_or_filebytes, "r") as zip_ref:
        # if a bids-style participants info file was included, read this in separately
        participants_tmp = [
            z for z in zip_ref.namelist() if os.path.basename(z) == "participants.tsv"
        ]
        participants_info = None

        if len(participants_tmp) > 0:
            participants_info = pd.read_table(
                zip_ref.open(participants_tmp[0]), delimiter="\t"
            )

        if participants_info is None:
            participants_list = [
                parse_bids_string(z)["participant_id"] for z in zip_ref.namelist()
            ]
            participants_list = np.unique(
                [s for s in participants_list if s is not None]
            )
            participants_info = pd.DataFrame({"participant_id": participants_list})

        totalfiles=len(zip_ref.namelist())
        pbar=tqdm(total=totalfiles, dynamic_ncols=True, leave=False, position=0)
        for zfile in zip_ref.namelist():
            if os.path.basename(zfile) == "participants.tsv":
                # skip participants info in main data loop
                continue
            if allowed_extensions_in_zip is not None:
                if not any(
                    [zfile.lower().endswith(ext) for ext in allowed_extensions_in_zip]
                ):
                    continue
            with zip_ref.open(zfile) as zfile_bytes:
                if zfile.lower().endswith(".csv"):
                    data_tmp = np.loadtxt(zfile_bytes, delimiter=",",comments=['#','!','%'])

                elif zfile.lower().endswith(".tsv"):
                    data_tmp = np.loadtxt(zfile_bytes, delimiter="\t",comments=['#','!','%'])

                elif zfile.lower().endswith(".mat"):
                    matdata = loadmat(zfile_bytes, simplify_cells=True)
                    matfields = ["data", "C", "SC", "FC"]
                    for m in matfields:
                        if m in matdata:
                            data_tmp = matdata[m]
                            break

                else:
                    # unrecognized file format in zip (could be a random OS file like .DS_Store)
                    continue

            bids_result = parse_bids_string(zfile)
            conntype = bids_result["inputtype"]
            subject = bids_result["participant_id"]
            if conntype not in conndata_squaremats:
                conndata_squaremats[conntype] = []
                conndata_participants[conntype] = []
            conndata_squaremats[conntype].append(data_tmp)
            conndata_participants[conntype].append(subject)
            pbar.update(1)
        pbar.close()
        
        # now reorder all conndata entries to the same subject order
        participants_info = participants_info.drop_duplicates(
            subset=["participant_id"]
        ).reset_index(drop=True)
        participants_list = participants_info["participant_id"].values

        conndata_subjidx = {}
        for conntype in conndata_squaremats:
            sidx = [
                np.where(np.array(conndata_participants[conntype]) == s)[0][0]
                for s in participants_list
            ]
            conndata_participants[conntype] = [
                conndata_participants[conntype][i] for i in sidx
            ]
            conndata_squaremats[conntype] = [
                conndata_squaremats[conntype][i] for i in sidx
            ]

    return conndata_squaremats, participants_info

def load_nemodata_zip(filename, filebytes=None, bidsify_subjects=True):
    # output: conndata_squaremats['conntype']=list([roi x roi])
    #         participants_info
    conndata_squaremats = {}
    conndata_participants = {}
    participants_info = None
    
    if filebytes is not None:
        filename_or_filebytes = filebytes
    else:
        filename_or_filebytes = filename
    
    with zipfile.ZipFile(filename_or_filebytes, "r") as zip_ref:
        #compile information about nemoSC files in .zip
        nemofiles=[f for f in zip_ref.namelist() if re.search(r'_nemo_output_.*nemoSC.*_mean\.mat$', f)]
        if len(nemofiles) == 0:
            raise Exception("No nemoSC files found in zip file")
        
        nemofiles_justfile=[os.path.split(f)[-1] for f in nemofiles]
        nemosubj=[re.sub(r'^(.+)_nemo_output_.+$',r'\1',f) for f in nemofiles_justfile]
        nemosubj_unique=list(set(nemosubj))
        
        nemofiles_pattern=list(set([re.sub(r'^(.+/)?(.+)(_nemo_output_.+$)',r'\1{SUBJECT}\3',f) for f in nemofiles]))
        
        nemofiles_pattern_flavor=[re.sub(r'.+nemo_output_(sdstream|ifod2act)_chacoconn_(.+)_nemoSC(_sift2)?(_volnorm)?_mean\.mat$',r'SC\1_\2\3\4',f) for f in nemofiles_pattern]
        nemofiles_pattern_flavor=[f.replace("_sift2_volnorm","_sift2volnorm") for f in nemofiles_pattern_flavor]
        nemofiles_pattern_flavor=[f if f.endswith("volnorm") else f+"_count" for f in nemofiles_pattern_flavor]
        nemofiles_pattern_flavor=[f.replace("_sift2_count","_sift2") for f in nemofiles_pattern_flavor]
        
        nemoconfig={}
        nemoconfig_file = [ z for z in zip_ref.namelist() if z.endswith("_config.json") ]
        if len(nemoconfig_file) > 0:
            with zip_ref.open(nemoconfig_file[0]) as jbytes:
                nemoconfig=json.load(jbytes)
        
        if 'siftweights' in nemoconfig and nemoconfig['siftweights']:
            if not all(["_sift2" in f for f in nemofiles_pattern_flavor]):
                #older nemo tool did not add "sift2" to output files, so check if config used sift weights and add this to flavor name if needed
                nemofiles_pattern_flavor=[f.replace("_volnorm","_sift2volnorm") for f in nemofiles_pattern_flavor]
                nemofiles_pattern_flavor=[f.replace("_count","_sift2") for f in nemofiles_pattern_flavor]
        
        #replace some of the atlas names (eg: fs86subj->fs86), or cocommpsuit439->coco439
        nemofiles_pattern_flavor=[f.replace('subj_','_') for f in nemofiles_pattern_flavor]
        nemofiles_pattern_flavor=[f.replace('_cocommpsuit439_','_coco439_') for f in nemofiles_pattern_flavor]
        
        print("")
        print("Flavors in nemo zip:")
        _=[print(f'{flav}={pat}') for flav,pat in zip(nemofiles_pattern_flavor,nemofiles_pattern)]
        
        # if a bids-style participants info file was included, read this in separately
        participants_tmp = [ z for z in zip_ref.namelist() if os.path.basename(z) == "participants.tsv" ]
        participants_info = None
        
        if len(participants_tmp) > 0:
            participants_info = pd.read_table(
                zip_ref.open(participants_tmp[0]), delimiter="\t"
            )
            if not 'subject' in participants_info:
                participants_info['subject']=[re.sub("^sub-","",s) for s in participants_info['participant_id']]
        
        if participants_info is None:
            subjsplit=['train' for s in nemosubj_unique]
            bids_subjects=[]
            for isubj,s in enumerate(nemosubj_unique):
                if re.match(r'sub-[A-Za-z0-9]+', s):
                    bids_subjects+=[s]
                else:
                    if bidsify_subjects:
                        bids_subjects+=['sub-' + re.sub(r'[^a-zA-Z0-9]', '', s)]
                    else:
                        bids_subjects+=['sub-%04d' % (isubj+1)]
            participants_info=pd.DataFrame({
                'participant_id':bids_subjects,
                'subject':nemosubj_unique, 
                'train_val_test':subjsplit
            })
        
        totalfiles=len(nemosubj_unique) * len(nemofiles_pattern_flavor)
        pbar=tqdm(total=totalfiles, dynamic_ncols=True, leave=False, position=0)
        
        for subject in nemosubj_unique:
            subjrow=participants_info[(participants_info['participant_id']==subject) | (participants_info['subject']==subject)]
            if len(subjrow)==0:
                raise Exception(f'Invalid participants.tsv. Subject {subject} not found.')
            bids_subj=subjrow['participant_id'].values[0]
            
            for conntype,flavpat in zip(nemofiles_pattern_flavor,nemofiles_pattern):
                f=flavpat.format(SUBJECT=subject)
                if not f in nemofiles:
                    print(f"Missing {flavpat} for subject {subject}")
                    continue
                with zip_ref.open(f) as zb:
                    if f.lower().endswith(".csv"):
                        data_tmp = np.loadtxt(zb, delimiter=",",comments=['#','!','%'])
                    
                    elif f.lower().endswith(".tsv"):
                        data_tmp = np.loadtxt(zb, delimiter="\t",comments=['#','!','%'])
                    
                    elif f.lower().endswith(".mat"):
                        matdata = loadmat(zb, simplify_cells=True)
                        matfields = ["data", "C", "SC", "FC"]
                        for m in matfields:
                            if m in matdata:
                                data_tmp = matdata[m]
                                break
                    else:
                        # unrecognized file format in zip (could be a random OS file like .DS_Store)
                        continue
                if not conntype in conndata_squaremats:
                    conndata_squaremats[conntype] = []
                    conndata_participants[conntype] = []
                conndata_squaremats[conntype].append(data_tmp)
                conndata_participants[conntype].append(bids_subj)
                pbar.update(1)
        pbar.close()
        
        # now reorder all conndata entries to the same subject order
        participants_info = participants_info.drop_duplicates(
            subset=["participant_id"]
        ).reset_index(drop=True)
        participants_list = participants_info["participant_id"].values
        
        for conntype in conndata_squaremats:
            if not all([s in conndata_participants[conntype] for s in participants_list]):
                raise Exception('All subjects must have the same set of output flavors')
            sidx = [
                np.where(np.array(conndata_participants[conntype]) == s)[0][0]
                for s in participants_list
            ]
            conndata_participants[conntype] = [
                conndata_participants[conntype][i] for i in sidx
            ]
            conndata_squaremats[conntype] = [
                conndata_squaremats[conntype][i] for i in sidx
            ]
    
    return conndata_squaremats, participants_info

def validate_data(conndata_squaremats, participants_info):
    # confirm that all data matrices for each conntype are the same size
    # and that there is one matrix per participant
    participants_list = participants_info["participant_id"].values
    conndata_subjectcount = {k: len(v) for k, v in conndata_squaremats.items()}
    conndata_matsize = {k: [m.shape for m in v] for k, v in conndata_squaremats.items()}

    conndata_matsize_template = {k: v[0] for k, v in conndata_matsize.items()}
    valid_subject_count = all(
        [v == len(participants_list) for k, v in conndata_subjectcount.items()]
    )
    valid_matsize = all(
        [
            [m == conndata_matsize_template[k] for m in v]
            for k, v in conndata_matsize.items()
        ]
    )

    valid = valid_subject_count and valid_matsize

    return valid, conndata_matsize_template


def humanize_filesize(size, binary=False):
    basesize = 1000.0
    if binary:
        basesize = 1024.0
    for unit in ["B", "KB", "MB", "GB"]:
        if size < basesize:
            break
        size /= basesize
    return "%.1f %s" % (size, unit)


def data_shape_string(data):
    if isinstance(data, np.ndarray):
        return "x".join(["%d" % (d) for d in data.shape])
    elif isinstance(data, list):
        return "%dx[%s]" % (len(data), data_shape_string(data[0]))
    else:
        import torch
        if isinstance(data, torch.Tensor):
            return "torch[%s]" % ("x".join(["%d" % (d) for d in data.shape]))
        else:
            raise Exception("Unknown data type")

def callback_load_and_process_data(
    filename,
    filebytes=None,
    allowed_extensions=None,
    variablename_loaddata="conndata_squaremats",
    variablename_participants="participants_info",
    globals_set=None
):

    print("Loading data from %s" % (filename))
    allowed_extensions = None  #! hack because its breaking the upload option
    conndata_squaremats, participants_info = load_data_zip(
        filename, filebytes=filebytes, allowed_extensions=allowed_extensions
    )

    isvalid, conndata_sizetemplate = validate_data(
        conndata_squaremats, participants_info
    )
    assert isvalid, "Data validation failed"

    if globals_set is not None:
        globals_set[variablename_loaddata] = conndata_squaremats
        globals_set[variablename_participants] = participants_info
    else:
        globals()[variablename_loaddata] = conndata_squaremats
        globals()[variablename_participants] = participants_info

    print(
        "Loaded data into '%s' and '%s'"
        % (variablename_loaddata, variablename_participants)
    )

    print("participants_info contains %d subjects" % (participants_info.shape[0]))
    if "train_val_test" in participants_info:
        train_count = participants_info[
            participants_info["train_val_test"] == "train"
        ].shape[0]
        print(
            "  %d/%d subjects are designated as 'training' for domain adaptation"
            % (train_count, participants_info.shape[0])
        )

    for conntype, v in conndata_squaremats.items():
        print(
            "input type %s has %d [%s] matrices"
            % (conntype, len(v), "x".join(["%d" % (d) for d in v[0].shape]))
        )


def callback_saveoutput(
    variablename,
    filename,
    bids_desc=None,
    variablename_participants="participants_info",
    globals_set=None
):
    if globals_set is not None:
        conndata_to_save = globals_set[variablename]
        participants_info = globals_set[variablename_participants]
    else:
        conndata_to_save = globals()[variablename]
        participants_info = globals()[variablename_participants]
    
    if filename.lower().endswith(".mat"):
        savemat(
            filename,
            {
                "predicted_alltypes": conndata_to_save,
                "subjects": participants_info["participant_id"].values,
            },
            format="5",
            do_compression=True,
            long_field_names=True
        )
    elif filename.lower().endswith(".zip"):
        save_data_zip(
            filename,
            conndata_to_save,
            participants_info,
            bids_desc=bids_desc,
            verbose=True,
        )
    else:
        raise ValueError("Unknown file extension for %s" % (filename))

    print(
        "Saved %s (%s)"
        % (os.path.abspath(filename), humanize_filesize(os.path.getsize(filename)))
    )

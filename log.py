import os
import sys
from datetime import datetime
import shutil

class Logger(object):
    """Helper class to make print() statements redirect to both stdout (screen) and a log file.
     Can also change the log file mid-script and it will move the old log to the new location and keep logging

    Example:
    ```
    log=Logger(logfile="logfile.txt",append=False)
    print("Just some log info")
    ```
    or: 
    ```
    log=Logger(logfile="auto",append=False) #start logging to a temp file
    print("I dont know what to call the log file yet because I havent read in the data")
    data=read(somefile) #read data that gives info about log file name
    log.transfer("newlogfile"+data.information()+".txt")
    print("Now I know what to call it")
    ```
    """
    def __init__(self,logfile=None,append=False):
        self.terminal = sys.stdout
        self.logfile=logfile
        self.logfile_append=append
        self.logfile_auto_prefix=None

        if self.logfile is not None:
            if logfile.lower()=="auto":
                self.logfile_auto_prefix="AUTOLOGTEMP"
                timestamp_suffix=datetime.now().strftime("%Y%m%d_%H%M%S.%f")
                self.logfile=os.path.join(os.curdir,self.logfile_auto_prefix+timestamp_suffix+".txt")
            if append:
                mode="a"
            else:
                mode="w"
            self.log = open(self.logfile, mode, buffering=1)
        #sys.stdout.reconfigure(line_buffering=True)
        sys.stdout=self

   
    def write(self, message):
        self.terminal.write(message)
        if self.logfile is not None:
            self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    

    def is_auto(self):
        return self.logfile_auto_prefix is not None and self.logfile.split(os.sep)[-1].startswith(self.logfile_auto_prefix)
    
    def transfer(self,newlogfile):
        """
        Rename log file and restart Logger with append. (Useful when we started logging to a temp filename)
        """
        #But just modify existing Logger object so it doesn't go out of scope
        self.log.close()
        shutil.move(self.logfile,newlogfile)
        self.logfile=newlogfile
        self.logfile_auto_prefix=None
        self.log = open(self.logfile,mode="a", buffering=1)
        return self
    
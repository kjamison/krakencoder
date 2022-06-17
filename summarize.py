import numpy as np
from scipy.io import loadmat
import sys

matfile=sys.argv[1]
maxnum=None
if len(sys.argv) > 2:
	maxnum=int(sys.argv[2])

def durationToString(numseconds):
    if numseconds < 60:
        return "%.3f seconds" % (numseconds)
    newms = numseconds % 1
    newseconds = int(numseconds) % 60
    newminutes = int(numseconds / 60) % 60
    newhours = int(numseconds / (60*60)) % 24
    newdays = int(numseconds / (60*60*24))
    newstring=""
    if newdays > 0:
        newstring+="%gd" % (newdays)
    if newhours > 0:
        newstring+="%gh" % (newhours)
    if newminutes > 0:
        newstring+="%gm" % (newminutes)
    if newms > 0:
        newstring+="%.3fs" % (newseconds+newms)
    elif newseconds > 0:
        newstring+="%gs" % (newseconds)
    return newstring

M=loadmat(matfile)
ep=M['current_epoch'][0,0]
finalep=M['nbepochs'][0,0]
eptimes=M['epoch_timestamp'][0,:]

print("training paths:")
[print(x) for x in M["trainpath_names"]]

epdiff=np.nanmedian(np.diff(eptimes))
eplatestdiff=np.nanmedian(np.diff(eptimes[ep-25:]))

print("Current epoch: %d" % (ep))
print("Median epoch/sec: %f" % (epdiff))
print("Last 25 epoch/sec: %f" % (eplatestdiff))
if ep == finalep-1:
    totalsec=np.round(eptimes[-1]-eptimes[0])
    print("Finished %d epochs in %dsec (%s)" % (finalep,totalsec,durationToString(totalsec)))
else:
    finalsec=np.round(finalep*eplatestdiff)
    remsec=np.round((finalep-(ep+1))*eplatestdiff)
    print("Expect to finish %d epoch in %dsec (%s). %dsec (%s) remaining." % (finalep,finalsec,durationToString(finalsec),remsec,durationToString(remsec)))

if maxnum and ep > maxnum:
	print("  Median epoch/sec up to %d: %f" % (maxnum,np.nanmedian(np.diff(eptimes[:maxnum]))))
	print("  Last 25 epoch/sec up to %d: %f" % (maxnum,np.nanmedian(np.diff(eptimes[(maxnum-25):maxnum]))))

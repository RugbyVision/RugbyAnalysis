import h5py
import numpy as np

path = "PlayerDetection/Resource/Arrays/RugbyGame_0.h5" # change as you like

# read hdf5 file
f = h5py.File(path, 'r')
# obtain the arrays in ndarray form
for i in range(345):
    print(np.array(f["Frame_" + str(i) + "_People"]))
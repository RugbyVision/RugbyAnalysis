import h5py
import numpy as np

path = "PlayerDetection/Resource/Arrays/RugbyGame_0.h5"

f = h5py.File(path, 'r')
for i in range(345):
    print(np.array(f["Frame_" + str(i) + "_People"]))
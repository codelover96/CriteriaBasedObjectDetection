# # Creating Train / Val / Test folders (One time use)
import os
import numpy as np
import shutil
from pathlib import Path

root_dir = Path("G:/airbus-ship-detection/txt")  # data root path

val_ratio = 0.25
test_ratio = 0.05

# Creating partitions of the data after shuffling

allFileNames = os.listdir(root_dir)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames) * (1 - (val_ratio + test_ratio))),
                                                           int(len(allFileNames) * (1 - test_ratio))])

os.makedirs(os.path.join(root_dir, "train"))
os.makedirs(os.path.join(root_dir, "val"))
os.makedirs(os.path.join(root_dir, "test"))

train_FileNames = [str(root_dir) + '\\' + name for name in train_FileNames.tolist()]
val_FileNames = [str(root_dir) + '\\' + name for name in val_FileNames.tolist()]
test_FileNames = [str(root_dir) + '\\' + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))

# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, os.path.join(root_dir, 'train'))

for name in val_FileNames:
    shutil.copy(name, os.path.join(root_dir, 'val'))

for name in test_FileNames:
    shutil.copy(name, os.path.join(root_dir, 'test'))
"""
OUTPUT FOR IMAGES

G:\Programs\anaconda3\envs\opencv\python.exe "G:/Programming/Python/Edge Detection/split-to-train-test-val.py"
Total images:  42556
Training:  29789
Validation:  10639
Testing:  2128

Process finished with exit code 0
"""

"""
OUTPUT FOR TXT
G:\Programs\anaconda3\envs\opencv\python.exe "G:/Programming/Python/Edge Detection/split-to-train-test-val.py"
Total images:  42556
Training:  29789
Validation:  10639
Testing:  2128

Process finished with exit code 0
"""
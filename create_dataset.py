import numpy as np
import pickle
from os import path

CUSTOM_DATASET_NAME = 'no-eyes-no-conf'
#Hello

keep = 1000
save = True
folder = './data'
remove_joints = [14,15] #LEye and REye
keep_indexes = [i for i in range(0,18) if i not in remove_joints]

origin_path = f'{folder}/Kinetics/kinetics-skeleton'
train_d = 'train_data.npy'
val_d = 'val_data.npy'
train_l = 'train_label.pkl'
val_l = 'val_label.pkl'
to_p = './' + CUSTOM_DATASET_NAME

train_data = np.load(path.join(origin_path,train_d), mmap_mode='r')
train_data = train_data[:keep]

val_data = np.load(path.join(origin_path,val_d), mmap_mode='r')
val_data = val_data[:keep]

# The shape is (100, 3, 300, 18, 2), we want (100, 2, 300, 18, 2) - where the second shape has the last value removed
def remove_confidence(narray):
    return narray[:, :2, :, :, :]

# We want to remove the joins specified in remove_joints
def remove_joints(narray):
    return narray[:, :, :,keep_indexes,:]

def do_both(narray):
    narray = remove_confidence(narray)
    narray = remove_joints(narray)
    return narray

train_data = do_both(train_data)
val_data = do_both(val_data)

with open(path.join(origin_path,train_l), 'rb') as pickle_file:
    train_label = pickle.load(pickle_file)
    train_label = [train_label[0][:keep], train_label[1][:keep]]

with open(path.join(origin_path,val_l), 'rb') as pickle_file:
    val_label = pickle.load(pickle_file)
    val_label = [val_label[0][:keep], val_label[1][:keep]]

if save:
    np.save(path.join(to_p, train_d), train_data)
    np.save(path.join(to_p, val_d), val_data)

    with open(path.join(to_p,train_l), 'wb') as pickle_file:
        pickle.dump(train_label, pickle_file)

    with open(path.join(to_p,val_l), 'wb') as pickle_file:
        pickle.dump(val_label, pickle_file)
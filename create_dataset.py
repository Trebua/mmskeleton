import numpy as np
import pickle
from os import path, mkdir
import argparse
from shutil import rmtree

# The shape is (100, 3, 300, 18, 2), we want (100, 2, 300, 18, 2) - where the second shape has the last value removed
def remove_confidence(narray):
    return narray[:, :2, :, :, :]

# We want to remove the joins specified in remove_joints
def remove_joints(narray):
    return narray[:, :, :,keep_indexes,:]

# Function for creating dataset
# name: specifies the folder name of the created dataset
# keep: specifies how many samples to keep. None => all
# remove_joint: removes joint specified in remove_joints in the function if True. Should probably always be eyes.
# remove_conf: If True removes the 3rd channel which contains confidence values
def create_dataset(name='custom_data', keep=None, remove_joint=True, remove_conf=True):
    print('Starting processing.')
    print(f'Remove joints: {remove_joint}')
    print(f'Remove confidence: {remove_conf}')
    data_folder = './data'
    to_remove = [14,15] #LEye and REye
    keep_indexes = [i for i in range(0,18) if i not in to_remove]

    origin_path = f'{data_folder}/Kinetics/kinetics-skeleton'
    train_d = 'train_data.npy'
    val_d = 'val_data.npy'
    train_l = 'train_label.pkl'
    val_l = 'val_label.pkl'
    to_p = path.join(data_folder,name)
    if path.exists(to_p):
        print(f'{to_p} already exists.')
        delete = input(f'Delete {to_p}? y/n: ')
        if delete == 'y':
            rmtree(to_p)
            print(f'{to_p} deleted. Will make a new one.')
        else:
            print('No selected, program exits.')
            return
    mkdir(to_p)

    print('\nLoading training data')
    train_data = np.load(path.join(origin_path,train_d), mmap_mode='r')
    train_data = train_data[:keep]
    print('Finished loading')

    print('\nLoading validation data')
    val_data = np.load(path.join(origin_path,val_d), mmap_mode='r')
    val_data = val_data[:keep]
    print('Finished loading')

    if remove_conf:
        print('\nRemoving confidence')
        train_data = remove_confidence(train_data)
        val_data = remove_confidence(val_data)
        print('No confidence left')
    if remove_joint:
        print('\nRemoving joints (eyes)')
        train_data = remove_joints(train_data)
        val_data = remove_confidence(val_data)
        print('Blindness acheived')

    print('\nLoading labels')
    with open(path.join(origin_path,train_l), 'rb') as pickle_file:
        train_label = pickle.load(pickle_file)
        train_label = [train_label[0][:keep], train_label[1][:keep]]

    with open(path.join(origin_path,val_l), 'rb') as pickle_file:
        val_label = pickle.load(pickle_file)
        val_label = [val_label[0][:keep], val_label[1][:keep]]
    print('Finished loading')

    print('\nSaving everything')
    np.save(path.join(to_p, train_d), train_data)
    np.save(path.join(to_p, val_d), val_data)

    with open(path.join(to_p,train_l), 'wb') as pickle_file:
        pickle.dump(train_label, pickle_file)

    with open(path.join(to_p,val_l), 'wb') as pickle_file:
        pickle.dump(val_label, pickle_file)
    print(f'Everything saved in {to_p}')

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Specifies the name of the new dataset folder")
parser.add_argument("-j", "--joints", help="1 removes joints, 0 keeps joints")
parser.add_argument("-c", "--confidence", help="1 removes eyes, 0 keeps eyes")
args = parser.parse_args()
name = args.name
remove_joint = True if int(args.joints) else False
remove_conf = True if int(args.confidence) else False
create_dataset(name=name, remove_joint=remove_joint, remove_conf=remove_conf)
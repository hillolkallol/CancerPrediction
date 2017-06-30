import numpy as np
import pandas as pd
import dicom
import os
import matplotlib.pyplot as plt
import cv2
import math

pixel_size = 20
slice_size = 20

tempFile = open('dataset-may-6-20-20-20-1500.txt', 'w')
tempFile.truncate()

tempFileCsv = open('dataset-may-6-20-20-20-1500.csv', 'w')
tempFileCsv.truncate()

def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def mean(a):
    return sum(a) / len(a)

def data_processing(patient, data_labels, img_size=20, img_slices=20, visualize=True):

    # Credit: Sentdex
    # Link: https://www.kaggle.com/sentdex/data-science-bowl-2017/first-pass-through-data-w-3d-convnet

    label = data_labels.get_value(patient, 'cancer')
    path = data_directory + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    # print(label)
    # plt.imshow(slices[120].pixel_array)
    # plt.show()

    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array), (img_size, img_size)) for each_slice in slices]

    for x in slices:
        for y in x:
            y[y < -1000] = -1000
            y[y > 300] = 0

    chunk_sizes = math.ceil(len(slices) / img_slices)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if len(new_slices) == img_slices - 1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == img_slices - 2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == img_slices + 2:
        new_val = list(map(mean, zip(*[new_slices[img_slices - 1], new_slices[img_slices], ])))
        del new_slices[img_slices]
        new_slices[img_slices - 1] = new_val

    if len(new_slices) == img_slices + 1:
        new_val = list(map(mean, zip(*[new_slices[img_slices - 1], new_slices[img_slices], ])))
        del new_slices[img_slices]
        new_slices[img_slices - 1] = new_val

    # new_slices = np.array(new_slices).astype(int)

    mean_value = np.mean(new_slices)
    std = np.std(new_slices)
    new_slices = new_slices - mean_value
    new_slices = new_slices / std

    if visualize:
        fig = plt.figure()
        for num, each_slice in enumerate(new_slices):
            y = fig.add_subplot(4, 5, num + 1)
            y.imshow(each_slice)
        plt.show()

    for x in new_slices:
        for y in x:
            for i in range(0, y.size):
                tempFile.write(str(y[i]))
                tempFile.write(', ')

                tempFileCsv.write(str(y[i]))
                tempFileCsv.write(', ')
            # tempFile.write('\n')
        tempFile.write(str(label))
        tempFileCsv.write(str(label))
    tempFile.write('\n')
    tempFileCsv.write('\n')

    if label == 1:
        label = np.array([0, 1])
    elif label == 0:
        label = np.array([1, 0])

    return np.array(new_slices), label

######################################################################
data_directory = 'cancer_sample_data/'
patients = os.listdir(data_directory)
labels_csv = pd.read_csv('stage1_labels.csv', index_col=0)

final_dataset = []
for x, patient in enumerate(patients[1:2]):
    print(x)
    try:
        img_data, label = data_processing(patient, labels_csv, img_size=pixel_size, img_slices=slice_size)
        # print(img_data.shape,label)

        # img_data = img_data.astype(int)
        # plot_3d(img_data)

        plt.hist(img_data.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units")
        plt.ylabel("Image Values")
        plt.show()

        final_dataset.append([img_data, label])
    except KeyError as e:
        print('This is unlabeled data!')
        pass

np.save('dataset-may-6-20-20-20-1500.npy', final_dataset)


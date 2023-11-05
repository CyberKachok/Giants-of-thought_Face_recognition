from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from os import listdir
import timeit
from collections import defaultdict


def list_files(directory, contains, remove_duplicates=False):
    if remove_duplicates:
        return list(f for f in listdir(directory) if contains in f and 'pgm' in f)
    return list(f for f in listdir(directory) if contains in f)


def create_dataframe(filenames):
    df = filenames[0].str.split(".", expand=True)
    df["filename"] = filenames

    df = df.rename(columns = {0:"subject", 1:"category"})
    df['subject'] = df.subject.str.replace('subject', '')
    df.apply(pd.to_numeric, errors='coerce').dropna()
    df['subject'] = pd.to_numeric(df["subject"])
    return df


def extract_image_data(df, directory, aggregate_type='None', w=243, h=320):
    '''
    создает np.array со всеми фотографиями и лейблами из директории

    '''

    if aggregate_type == 'None':
        labels = []
        dataset = np.zeros((w, h, df.shape[0]))
        for i in range(df.shape[0]):
            dataset[:, :, i] = plt.imread(directory + df.iloc[i, 3])
            labels.append(df.iloc[i, 3])
    elif aggregate_type == 'Mean':
        dataset = np.zeros((w, h, df['subject'].nunique()))
        labels = []
        for i in range(df.shape[0]):
            dataset[:, :, df.iloc[i, 0] - 1] += plt.imread(directory + df.iloc[i, 3])
            label = df.filename.str.split(".", expand=True)[0]
            #if label not in labels:
            #labels.append(label)
        labels = list(np.unique(df.filename.str.split(".", expand=True)[0]))
        dataset = dataset / 11
    elif aggregate_type == 'SVD':
        dataset = np.zeros((w * h, 10, df['subject'].nunique()))
        counter = 0
        for i in range(df.shape[0]):
            if counter > 9:
                counter = 0
                continue
            #print(counter, df.iloc[i, 3])
            dataset[:, counter, df.iloc[i, 0] - 1] = plt.imread(directory + df.iloc[i, 3]).flatten(order='F')

            counter += 1

        labels = list(np.unique(df.filename.str.split(".", expand=True)[0]))

    return dataset, np.array(labels)


def find_majority(labels):
    '''
    для функции predict, находит наиболее частый элемент

    '''
    counter = defaultdict(int)
    for label in labels:
        counter[label] += 1

    # Finding the majority class.
    majority_count = max(counter.values())
    for key, value in counter.items():
        if value == majority_count:
            return key


def predict(data, labels, target, target_label, k):
    '''ранжирует k первых категорий по расстоянию и берет наиболее частую'''
    target_vector = target.flatten(order='F')
    target_name = target_label.split('.')[0]

    distances = []
    for i in range(data.shape[2]):
        database_vector = data[:, :, i].flatten(order='F')
        database_label = labels[i].split('.')[0]
        distances.append((np.linalg.norm(target_vector - database_vector), database_label))

    sorted_distances = sorted(distances, key=lambda vals: vals[0])
    k_labels = [label for (_, label) in sorted_distances[:k]]
    #print(k_labels)
    return find_majority(k_labels)


def test_predictor(predictor, data, labels, **kwargs):
    '''
    надо либо доделать либо выкинуть
    '''
    counter = 0
    for i in range(data.shape[2]):
        #print(i)
        target = data[:, :, i]
        target_label = labels[i]
        train_data = np.delete(data, i, axis=2)
        train_labels = np.delete(labels, i)
        vector_result = predictor(data=train_data, labels=train_labels, target=target, target_label=target_label, **kwargs)
        if vector_result == target_label.split('.')[0]:
            #print(vector_result, ': True answer')
            counter += 1
    return counter / data.shape[2]

def predict_svd(data, labels, target, target_label, k):
    '''
    ранжирует k первых категорий по расстоянию и берет наиболее частую
    '''

    target_vector = target

    norm_target = np.linalg.norm(target_vector)
    target_name = target_label.split('.')[0]

    distances = []
    for i in range(data.shape[2]):
        database_matrix = data[:, :, i]
        database_label = labels[i].split('.')[0]
        distances.append((np.linalg.norm((target_vector - database_matrix @ (database_matrix.T @ target_vector)) / norm_target), database_label))

    sorted_distances = sorted(distances, key=lambda vals: vals[0])
    k_labels = [label for (_, label) in sorted_distances[:k]]
    #print(k_labels)
    return find_majority(k_labels)
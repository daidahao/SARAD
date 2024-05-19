import os
import time
import numpy as np

def save_scores(labels, scores, threshold, \
         model, dataset, folder, \
         timestamp=None,
    ):
    
    folder = os.path.join(folder, model)
    os.makedirs(folder, exist_ok=True)

    folder = os.path.join(folder, dataset)
    os.makedirs(folder, exist_ok=True)

    if timestamp is None:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    folder = os.path.join(folder, timestamp)
    os.makedirs(folder, exist_ok=True)
    
    path = os.path.join(folder, 'labels.csv')
    np.savetxt(path, labels, delimiter=',', fmt='%d')

    path = os.path.join(folder, 'scores.csv')
    np.savetxt(path, scores, delimiter=',')

    path = os.path.join(folder, 'threshold.txt')
    with open(path, 'w') as f:
        f.write(str(threshold))
    
    return timestamp

def save_diagno(labels, diagno, \
         model, dataset, folder, \
         timestamp=None,
    ):
    
    folder = os.path.join(folder, model)
    os.makedirs(folder, exist_ok=True)

    folder = os.path.join(folder, dataset)
    os.makedirs(folder, exist_ok=True)

    if timestamp is None:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    folder = os.path.join(folder, timestamp)
    os.makedirs(folder, exist_ok=True)
    
    path = os.path.join(folder, 'labels.csv')
    np.savetxt(path, labels, delimiter=',', fmt='%d')

    path = os.path.join(folder, 'diagno.csv')
    np.savetxt(path, diagno, delimiter=',')

    return timestamp

def read_all_diagno(model, dataset, folder_):
    folder = os.path.join(folder_, model, dataset)
    for timestamp in sorted(os.listdir(folder))[::-1]:
        subfolder = os.path.join(folder, timestamp)
        if not os.path.isdir(subfolder):
            continue
        yield timestamp, *read_diagno(model, dataset, timestamp, folder_)


def read_diagno(model, dataset, timestamp, folder):
    folder = os.path.join(folder, model, dataset, timestamp)

    path = os.path.join(folder, 'labels.csv')
    labels = np.loadtxt(path, delimiter=',', dtype=int)

    path = os.path.join(folder, 'diagno.csv')
    diagno = np.loadtxt(path, delimiter=',')

    return labels, diagno


def read_all_scores(model, dataset, folder_):
    folder = os.path.join(folder_, model, dataset)
    for timestamp in os.listdir(folder):
        subfolder = os.path.join(folder, timestamp)
        if not os.path.isdir(subfolder):
            continue
        yield timestamp, *read_scores(model, dataset, timestamp, folder_)


def read_scores(model, dataset, timestamp, folder):
    folder = os.path.join(folder, model, dataset, timestamp)

    path = os.path.join(folder, 'labels.csv')
    labels = np.loadtxt(path, delimiter=',', dtype=int)

    path = os.path.join(folder, 'scores.csv')
    scores = np.loadtxt(path, delimiter=',')

    path = os.path.join(folder, 'threshold.txt')
    with open(path, 'r') as f:
        threshold = float(f.read())

    return labels, scores, threshold



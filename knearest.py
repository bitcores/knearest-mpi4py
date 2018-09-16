from mpi4py import MPI
import numpy as np
import math
import warnings
from collections import Counter
import pandas as pd
import random

comm = MPI.COMM_WORLD
mpirank = comm.rank
mpisize = comm.size

# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]
#########

def k_nearest_neighbors(data, predict, k=5):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclid = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclid, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
#   confidence level of the vote
#   confidence = Counter(votes).most_common(1)[0][1] / float(k)

    return vote_result


if mpirank == 0:
#   Dataset from http://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
    df = pd.read_csv("EEG-Eye-State.data")
#   no incomplete data we need to deal with, no id column
#   df.replace('?', -99999, inplace=True)
#   df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()

if mpirank == 0:
    random.shuffle(full_data)

    test_size = 0.2
    train_set = {0:[], 1:[]}
    test_scat = []
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    #test_data = np.array_split(test_data, mpisize)
    test_scat = list(chunks(test_data, int(math.ceil(len(test_data)/float(mpisize)))))
    #print test_scat
else:
    train_set = None
    test_scat = None

train_set = comm.bcast(train_set, root=0)
test_scat = comm.scatter(test_scat, root=0)

test_set = {0:[], 1:[]}
for i in test_scat:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

correct = comm.reduce(correct, op=MPI.SUM, root=0)
total = comm.reduce(total, op=MPI.SUM, root=0)

if mpirank == 0:
    print 'Accuracy:', float(correct)/total
    print correct, 'correct of', total

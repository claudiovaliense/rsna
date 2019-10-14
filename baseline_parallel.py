"""
Author: Claudio Moises Valiense de Andrade and NAMES PEOPLE GROUP RSNA UFF
Code objective: Kaggle Competition RSNA
"""
import numpy as np
from skimage.segmentation import (morphological_chan_vese, morphological_geodesic_active_contour)
from skimage.segmentation import (inverse_gaussian_gradient, checkerboard_level_set)
from skimage import io  # Load image file
from sklearn.neighbors import KNeighborsClassifier  # Classificador knn
import claudio_funcoes as cv
from sklearn.model_selection import train_test_split
from sklearn import metrics  # Checar a acuracia
import lib
import multiprocessing as mp  # Multiprocessing set train
import threading  # Multiprocessing set train
import queue


# Method Snake
def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


def process_file(filename):
    """Process contour method"""
    image = io.imread(filename)
    init_ls = checkerboard_level_set(image.shape, 5)
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(image, iterations, init_level_set=init_ls, smoothing=1, iter_callback=callback)
    return evolution[-1]


def processing_thread(file,label):
    count = 0
    file_name = dir + file
    #contour = process_file(file_name)  # method snake

    image = lib.read_image(file_name)  # feature hematoma, utilize hu
    hematoma = lib.substance_interval(image, 40, 90)

    #data.append(contour.flatten())
    data.append(hematoma.flatten())
    # data.append(np.append(contour.flatten(),hematoma.flatten()))
    target.append(label)

    return [data, target]

    '''count += 1
    if count == amount_files:
        break
    '''

def data_target(dir, label):
    """Process data and attribuited label"""
    count = 0
    data = []
    target = []

    que = queue.Queue()
    threads_list = list()

    for file in cv.list_files(dir):
        for i in range(mp.cpu_count()):
            t = threading.Thread(target=lambda q, arg1: q.put(processing_thread(arg1)), args=(que, file,label))
            t.start()
            threads_list.append(t)

    # Join all the threads
    for t in threads_list:
        t.join()

    list_geral = []
    # Check thread's return value
    while not que.empty():
        result = que.get()
        list_geral.append(result)


return [data, target]

# ----------- Main
dir_epidural = "//home/usuario/projetos/github/rsna/dataset/epidural/"
dir_normal = "//home/usuario/projetos/github/rsna/dataset/normal/"
k = 5  # k of knn classifier
data = []
target = []
amount_files = 100
iterations = 35  # method snake

datas, targets = data_target(dir_epidural, 'epidural')
for d in datas:
    data.append(d)
for t in targets:
    target.append(t)

datas, targets = data_target(dir_normal, 'normal')
for d in datas:
    data.append(d)
for t in targets:
    target.append(t)

X = np.array(data)
print("Data matrix size : {:.2f}MB".format(X.nbytes / (1024 * 1000.0)))
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=cv.AMOUNT_TEST, random_state=cv.SEED_RANDOM)

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print('k: ', k)
print('Accuracy: ', accuracy)

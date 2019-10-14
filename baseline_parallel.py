"""
Author: Claudio Moises Valiense de Andrade and NAMES PEOPLE GROUP RSNA UFF
Code objective: Kaggle Competition RSNA
"""
import numpy as np
from skimage.segmentation import (morphological_chan_vese, morphological_geodesic_active_contour)
from skimage.segmentation import (inverse_gaussian_gradient, checkerboard_level_set)
from skimage import io  # Load image file
from sklearn.neighbors import KNeighborsClassifier  # Classificador knn
from sklearn import svm # Classificador SVN
import claudio_funcoes as cv
from sklearn.model_selection import train_test_split
from sklearn import metrics  # Checar a acuracia
import lib
import multiprocessing as mp  # Multiprocessing set train
import threading  # Multiprocessing set train
import queue
import timeit  # calcular metrica de tempo



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


def processing_thread(file_name, label):
    contour = process_file(file_name)  # method snake

    #image = lib.read_image(file_name)  # feature hematoma, utilize hu
    #hematoma = lib.substance_interval(image, 30, 90)

    # combined method
    #return [hematoma.flatten()], [label]
    #return [np.append(contour.flatten(), hematoma.flatten())], [label]
    return [contour.flatten()], [label]


def data_target(dir, label):
    """Process data and attribuited label"""
    count = 0
    data = []
    data_geral = []
    target = []
    target_geral = []
    n_cores = mp.cpu_count()

    que = queue.Queue()
    threads_list = list()

    files = cv.list_files(dir)
    size_files = len(files)
    size_files = amount_files
    for a in range(size_files):
        for i in range(n_cores):
            if(count<size_files):
                t = threading.Thread(target=lambda q, arg1, arg2: q.put(processing_thread(arg1, arg2)), args=(que, dir+files[count], label))
                count += 1
                t.start()
                threads_list.append(t)
            else:
                break

        # Join all the threads
        for t in threads_list:
            t.join()


        # Check thread's return value
        while not que.empty():
            data, target = que.get()
            data_geral.append(data[0])
            target_geral.append(target[0])

    return [data_geral,target_geral]

# ----------- Main
#dir_epidural = "//home/usuario/projetos/github/rsna/dataset/epidural/"
#dir_normal = "//home/usuario/projetos/github/rsna/dataset/normal/"
dir_epidural ="//home/claudiovaliense/kaggle/rsna/epidural/"
dir_normal ="//home/claudiovaliense/kaggle/rsna/normal/"
k = 10  # k of knn classifier
data = []
target = []
amount_files = 200
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

#model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
#print('k: ', k)

model = svm.SVC(kernel='rbf', gamma='scale')

print("Train model")
ini = timeit.default_timer()
model.fit(X_train, y_train)
print("Time train model: %f" % (timeit.default_timer() - ini))

y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: ', accuracy)

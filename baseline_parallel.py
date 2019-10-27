"""
Author: Claudio Moises Valiense de Andrade and NAMES PEOPLE GROUP RSNA UFF
Code objective: Kaggle Competition RSNA
"""
import numpy as np
from skimage.segmentation import (morphological_chan_vese, morphological_geodesic_active_contour)
from skimage.segmentation import (inverse_gaussian_gradient, checkerboard_level_set)
from skimage import io  # Load image file
from sklearn.neighbors import KNeighborsClassifier  # Classificador knn
from sklearn import svm  # Classificador SVN
import claudio_funcoes as cv
from sklearn.model_selection import train_test_split
from sklearn import metrics  # Checar a acuracia
import lib
import multiprocessing as mp  # Multiprocessing set train
import threading  # Multiprocessing set train
import queue
import timeit  # calcular metrica de tempo
import sklearn.preprocessing as pre # utiliza normalize
import skimage
from skimage.filters import threshold_multiotsu
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from skimage.feature import canny # bord detect
from skimage.transform import hough_circle, hough_circle_peaks
from skimage import data, color
from skimage.draw import circle_perimeter

from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

#import cv2
from skimage.morphology import disk



AMOUNT_TEST = 0.2
SEED_RANDOM = 4


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

def snake_GAC(filename):
    image = io.imread(filename)
    gimage = inverse_gaussian_gradient(image)

    # Initial level set
    init_ls = np.zeros(image.shape, dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1
    # List with intermediate results for plotting the evolution
    evolution2 = []
    callback = store_evolution_in(evolution2)
    ls = morphological_geodesic_active_contour(gimage, iterations,
                                               iter_callback=callback)
    return evolution2[-1]



def processing_thread(dir, files, label, id_core):
    features = []
    labels = []
    for file_name in files:
        norm = 'max'
        image = lib.read_image(dir + file_name)  # feature hematoma, utilize hu
        #lib.plot('original', image)
        selem = disk(1)
        image = skimage.morphology.dilation(image, selem)
        #lib.plot('dilation', image)

        # features
        snake = process_file(dir + file_name)  # method snake
        hematoma = pre.normalize(lib.substance_interval(image, 30, 90), norm=norm)
        white_matter = pre.normalize(lib.substance_interval(image, 20, 30), norm=norm)
        ventriculo = pre.normalize(lib.substance_interval(image, 0, 15), norm=norm)
        white_tophat = pre.normalize(skimage.morphology.white_tophat(image), norm=norm)



        #lib.plot('eroted', eroded)



        '''edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)
        hough_radii = np.arange(1, 10, 1)
        hough_res = hough_circle(edges, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,total_num_peaks=3)
        # Draw them
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        image = color.gray2rgb(image)
        color.rgb2gray(image)
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
            image[circy, circx] = (220, 20, 20)

        ax.imshow(image, cmap=plt.cm.gray)
        plt.show()

        lib.plot('sem rgb', color.rgb2gray(image))
        '''


        #plt.imshow(white_tophat, cmap=plt.cm.bone)
        #plt.title('white_tophat')
        #plt.show()

        #hematoma =lib.substance_interval(image, 30, 90)
        #white_matter = lib.substance_interval(image, 20, 30)

        # resultado ruim se adicionar. Teste em 80 imagens

       # snake_method2 = snake_GAC(dir + file_name)
      	# bone =  pre.normalize(lib.substance_interval(image, 700, 3000), norm=norm)
       	# blood = pre.normalize(lib.substance_interval(image, 45, 65), norm=norm)

        #print("aqui1")
        #multiotsu = threshold_multiotsu(image, classes=5) # melhora com o aumento de classe, mas piorou ao adicionar outras features
        #print("aqui")
        #regions_multi = pre.normalize(np.digitize(image, bins=multiotsu), norm=norm)

        '''  colorized, otsu, thresholds = lib.multiotsu(image, 3)
          mask_ossos = np.zeros((512, 512))
          mask_ossos[otsu == 2] = 1
          ossos = image * mask_ossos
          ossos = pre.normalize(ossos, norm=norm)'''


        #con_hem = np.append(snake,hematoma)
        #con_hem =  np.append(con_hem,white_matter)
        con_hem = snake + hematoma + white_matter + ventriculo + white_tophat
        #lib.plot('combinacao', con_hem)
        #con_hem = pre.normalize(con_hem, norm=norm) # Ruim normalizar no final


        features.append(con_hem.flatten())
        #features.append(con_hem)
        #label = id_core % 5
        labels.append(label)
        cv.calculate_process(amount_files*2)

    return id_core, features, labels

    # combined method
    # return [hematoma.flatten()], [label]
    # return [np.append(contour.flatten(), hematoma.flatten())], [label]



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
    files = files[0:amount_files]
    files = cv.n_list(files, n_cores)

    for id_core in range(n_cores):
        t = threading.Thread(target=lambda q, dir, arg1, arg2, arg3: q.put(processing_thread(dir, arg1, arg2, arg3)),
                             args=(que, dir, files[id_core], label, id_core))
        t.start()
        threads_list.append(t)

    # Join all the threads
    for t in threads_list:
        t.join()

    return_cores = dict()
    # Check thread's return value
    while not que.empty():
        id_core, datas, targets = que.get()
        return_cores[id_core] = datas, targets

    # Ordenar na mesma ordem que pegou os arquivos
    for id in range(n_cores):
        datas, targets = return_cores[id]
        for data in datas:
            data_geral.append(data)
        for target in targets:
            target_geral.append(target)

    return [data_geral, target_geral]


# ----------- Main
dir_epidural = "../epidural/"
dir_normal = "../normal/"
# dir_epidural ="//home/claudiovaliense/kaggle/rsna/epidural/"
# dir_normal ="//home/claudiovaliense/kaggle/rsna/normal/"
k = 3  # k of knn classifier
data = []
target = []
amount_files = 40
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
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=AMOUNT_TEST, random_state=SEED_RANDOM)

model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
#model = RandomForestClassifier()
#model = svm.SVC(kernel='rbf', gamma='scale')
# print('k: ', k)



print("Train model")
ini = timeit.default_timer()



model.fit(X_train, y_train)
print("Time train model: %f" % (timeit.default_timer() - ini))

#y_prob = model.predict_proba(X_test)
#print('probaa: ', y_prob)

y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
#accuracy_prob = metrics.accuracy_score(y_test, y_prob)

print('Accuracy: ', accuracy)
#print('Accuracy prob: ', accuracy_prob)

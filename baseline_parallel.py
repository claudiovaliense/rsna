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
import id_label
import claudio_funcoes as cv
from sklearn.model_selection import train_test_split
from sklearn import metrics  # Checar a acuracia
import lib
import multiprocessing as mp  # Multiprocessing set train
import threading  # Multiprocessing set train
import queue
import timeit  # calcular metrica de tempo
import sklearn.preprocessing as pre  # utiliza normalize
import skimage
import pickle
import joblib
from skimage.filters import threshold_multiotsu
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from skimage.feature import canny  # bord detect
from skimage.transform import hough_circle, hough_circle_peaks
from skimage import data, color
from skimage.draw import circle_perimeter
from sklearn.preprocessing import MinMaxScaler

from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

# import cv2
from skimage.morphology import disk
from sklearn.preprocessing import MultiLabelBinarizer

import multiprocessing  # Version parallel

# id_label = id_label.return_id_label()
# cv.save_dict_file('id_label', id_label)
id_label = cv.load_dict_file('../id_label')

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
        # scaler = MinMaxScaler() # example minmax, scaler
        # hematoma = scaler.fit_transform(hematoma)

        norm = 'max'
        image = lib.read_image(dir + file_name)  # feature hematoma, utilize hu
        # print("image matrix size : {:.2f}MB".format(image.nbytes / (1024 * 1000.0)))
        # lib.plot('original', image)
        # selem = disk(1)
        # image = skimage.morphology.dilation(image, selem)
        # lib.plot('dilation', image)

        # features
        snake = process_file(dir + file_name)  # method snake
        hematoma = pre.normalize(lib.substance_interval(image, 30, 90), norm=norm).astype('float16')
        white_matter = pre.normalize(lib.substance_interval(image, 20, 30), norm=norm).astype('float16')
        ventriculo = pre.normalize(lib.substance_interval(image, 0, 15), norm=norm).astype('float16')
        white_tophat = pre.normalize(skimage.morphology.white_tophat(image), norm=norm).astype('float16')
        blood = pre.normalize(lib.substance_interval(image, 45, 65), norm=norm).astype('float16')

        # x = np.concatenate((snake,hematoma,white_matter,ventriculo,white_tophat,blood))
        # print("XXData matrix size : {:.2f}MB".format(x.nbytes / (1024 * 1000.0)))

        # con_hem = hematoma +  ventriculo + white_tophat + blood
        # print("Data matrix size : {:.2f}MB".format(con_hem.nbytes / (1024 * 1000.0)))
        # con_hem = hematoma + white_matter + ventriculo
        # print("Data matrix size : {:.2f}MB".format(con_hem.nbytes / (1024 * 1000.0)))


        np.savez_compressed(label + 'features/snake/' + file_name, snake=snake)
        np.savez_compressed(label + 'features/hematoma/' + file_name, hematoma=hematoma)
        np.savez_compressed(label + 'features/white_matter/' + file_name, white_matter=white_matter)
        np.savez_compressed(label + 'features/ventriculo/' + file_name, ventriculo=ventriculo)
        np.savez_compressed(label + 'features/white_tophat/' + file_name, white_tophat=white_tophat)
        np.savez_compressed(label + 'features/blood/' + file_name, blood=blood)

        cv.calculate_process((amount_files) / n_cores)

        # lib.plot('eroted', eroded)

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

        # plt.imshow(white_tophat, cmap=plt.cm.bone)
        # plt.title('white_tophat')
        # plt.show()

        # hematoma =lib.substance_interval(image, 30, 90)
        # white_matter = lib.substance_interval(image, 20, 30)

        # resultado ruim se adicionar. Teste em 80 imagens

        # snake_method2 = snake_GAC(dir + file_name)
        # bone =  pre.normalize(lib.substance_interval(image, 700, 3000), norm=norm)

        # print("aqui1")
        # multiotsu = threshold_multiotsu(image, classes=5) # melhora com o aumento de classe, mas piorou ao adicionar outras features
        # print("aqui")
        # regions_multi = pre.normalize(np.digitize(image, bins=multiotsu), norm=norm)

        '''  colorized, otsu, thresholds = lib.multiotsu(image, 3)
          mask_ossos = np.zeros((512, 512))
          mask_ossos[otsu == 2] = 1
          ossos = image * mask_ossos
          ossos = pre.normalize(ossos, norm=norm)'''

        # con_hem = np.append(snake,hematoma)
        # con_hem =  np.append(con_hem,white_matter)
        '''con_hem = snake + hematoma + white_matter + ventriculo + white_tophat + blood

        # lib.plot('combinacao', con_hem)
        # con_hem = pre.normalize(con_hem, norm=norm) # Ruim normalizar no final

        features.append(con_hem.flatten())
        # features.append(con_hem)
        # label = id_core % 5
        if label != 'testes':  # test model
            y = id_label[file_name].values()
            # aa = MultiLabelBinarizer().fit_transform(id_label[file_name].values())
            labels.append(np.array(list(y)).flatten())  # transform dict values in array

        

    return id_core, features, labels'''

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

    que = queue.Queue()
    threads_list = list()

    files = cv.list_files(dir)
    # files = files[0:amount_files]
    files = cv.n_list(files, n_cores)

    for id_core in range(n_cores):
        t = multiprocessing.Process(target=processing_thread, args=(dir, files[id_core], label, id_core))
        '''t = threading.Thread(
            target=lambda q, dir, arg1, arg2, arg3: q.put(processing_thread(dir, arg1, arg2, arg3)),
            args=(que, dir, files[id_core], label, id_core))'''
        t.start()
        threads_list.append(t)

    # Join all the threads
    for t in threads_list:
        t.join()

    '''return_cores = dict()
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

    return [data_geral, target_geral]'''


def load_parallel(files, id_label, path, test_model):
    process_list = list()
    X=[]
    Y=[]
    files = cv.n_list(files, n_cores)

    for id_core in range(n_cores):
        p = multiprocessing.Process(target=load_X_compress_parallel, args=(files[id_core], id_label, path, test_model, id_core))
        p.start()
        process_list.append(p)

    # Join all the threads
    for p in process_list:
        p.join()

    for i in range(n_cores):        
        X_core, Y_core = (return_process_dict[i])
        X.append(X_core)
        if test_model == True:
            Y.append(Y_core)

    # resultado dos cores na mesma ordem do ids de entradas
    data=[]
    labels=[]
    for core in range(n_cores):
        for index in range(len(X[core])):
            data.append(X[core][index])
            if test_model == True:
                labels.append(Y[core][index])

    return data, labels


def load_X_compress_parallel(files, id_label, path, test_model, id_core):
    features = []
    labels = []
    for file_name in files:
        snake = np.load(path+'features/snake/' + file_name+'.npz')['snake'].astype('int')
        blood = np.load(path+'features/blood/' + file_name+'.npz')['blood'].astype('float16')
        hematoma = np.load(path+'features/hematoma/' + file_name+'.npz')['hematoma'].astype('float16')
        ventriculo = np.load(path+'features/ventriculo/' + file_name+'.npz')['ventriculo'].astype('float16')
        white_matter = np.load(path+'features/white_matter/' + file_name+'.npz')['white_matter'].astype('float16')
        white_tophat = np.load(path+'features/white_tophat/' + file_name+'.npz')['white_tophat'].astype('float16')
        #con_hem = hematoma + white_matter + ventriculo + white_tophat + blood  # combined
        con_hem = snake + hematoma + white_matter + ventriculo + white_tophat + blood  # combined
        
        
        if len(con_hem.flatten())!= 262144:
            con_hem =  np.zeros(262144)
            print('tamanho', len(con_hem.flatten()))
           
        features.append(con_hem.flatten())

        if test_model == True:
            y = id_label[file_name].values()
            labels.append(np.array(list(y)).flatten().astype(('int')))  # transform dict values in ar

    return_process_dict[id_core] = features, labels


# ----------- Main
dir_train = "../dataset/stage_1_train_images/"
dir_test = "../dataset/stage_1_test_images/"
manager = multiprocessing.Manager()  #parallel
return_process_dict = manager.dict() #parallel


k = 3  # k of knn classifier
data = []
target = []
iterations = 35  # method snake
files_test = cv.list_files(dir_test)
#files_train = cv.list_files(dir_train)
#files_train = files_train[0:50000]
#files_test = files_test[0:1000]

#amount_files_train = len(files_train)
amount_files_test = len(files_test)
#amount_files = amount_files_test
n_cores = mp.cpu_count()

#data_target(dir_train, '')  # features in files
#data_target(dir_test, "teste/") # rodar no server ainda

ini = timeit.default_timer()


#X_train, Y_train = cv.load_X_compress(files_train, id_label, '', True)
#X_test, Y_test = cv.load_X_compress(files_test, id_label, 'teste/', True)
print('carregar treino')
#X_train, Y_train = load_parallel(files_train, id_label, '', True)
return_process_dict = manager.dict()
print("Carregar dados: %f" % (timeit.default_timer() - ini))


# '''
# format classifier
#X_train = np.array(X_train).astype('float16')
#Y_train = np.array(Y_train).astype('float16')



#Y_test = np.array(Y_test)  # teste model

#print("Data matrix size : {:.2f}MB".format(X_train.nbytes / (1024 * 1000.0)))
# X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=AMOUNT_TEST, random_state=SEED_RANDOM)

#model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
#model = RandomForestClassifier(n_estimators=10, random_state=SEED_RANDOM, n_jobs=-1)
# model = svm.SVC(kernel='rbf', gamma='scale')

print("Train model")
ini = timeit.default_timer()
#model.fit(X_train, Y_train)
print("Time train model: %f" % (timeit.default_timer() - ini))

# save the model to disk
#pickle.dump(model, open(cv.name_out('./KNN.model'), 'wb'))
#joblib.dump(model, open(cv.name_out('./RandomForest.model'), 'wb'))

model = joblib.load('RandomForest.50000')

#Y_pred = model.predict(X_test)
#accuracy = metrics.accuracy_score(Y_test, Y_pred)
#print('Accuracy: ', accuracy)

# amount_files_test = 78545
# imprime todas as probabilidades das classes por documento
# save result in file
with open(cv.name_out('./final_result.csv'), 'w', newline='') as csvfile:                    
        files = cv.n_list(files_test,2)
        print('len files: ', len(files[0]))
        csvfile.write('ID,Label\n')

        for i in range(2):
            files_test=files[i]
            X_test, Y_test = load_parallel(files_test, id_label, 'teste/', False) # test model, alter True           
           # print('X_test: ', X_test )
            X_test = np.array(X_test).astype('float16')
           # print('X_test: ', X_test )

            Y_prob = model.predict_proba(X_test)


            for doc in range(len(files_test)):
                for classe in range(6):
                    if classe == 0:
                        csvfile.write(str(files_test[doc]).split('.')[0] + '_epidural,' + str(Y_prob[classe][doc][1]) + '\n')
                    elif classe == 1:
                        csvfile.write(str(files_test[doc]).split('.')[0] + '_intraparenchymal,' + str(Y_prob[classe][doc][1]) + '\n')
                    elif classe == 2:
                        csvfile.write(str(files_test[doc]).split('.')[0] + '_intraventricular,' + str(Y_prob[classe][doc][1]) + '\n')
                    elif classe == 3:
                        csvfile.write( str(files_test[doc]).split('.')[0] + '_subarachnoid,' + str(Y_prob[classe][doc][1]) + '\n')
                    elif classe == 4:
                        csvfile.write(str(files_test[doc]).split('.')[0] + '_subdural,' + str(Y_prob[classe][doc][1]) + '\n')
                    elif classe == 5:
                        csvfile.write(str(files_test[doc]).split('.')[0] + '_any,' + str(Y_prob[classe][doc][1]) + '\n')


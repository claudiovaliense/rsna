"""
Autor: Claudio Moisés Valiense de Andrade
Objetivo: Criar um biblioteca de funções de uso geral
"""
import csv  # Manipular csv
import numpy  # Manipular arrau numpy
from sklearn.decomposition import LatentDirichletAllocation  # LDA
import timeit  # calcular metrica de tempo
import pickle  # Usado para salvar modelo
from sklearn.feature_extraction.text import TfidfVectorizer  # Utilizar metodo Tfidf
import os  # Variable in system
from datetime import datetime  # Datetime for time in file
from heapq import nlargest  # Order vector
import json  # Manipulate extension json

AMOUNT_TEST=0.2
SEED_RANDOM=4


def selecionar_ids_arquivo(file_ids_total, files_ids_selecionados):
    """
    Objetivo: Selecionar linhas de arquivo a partir de outro arquivo com ids.

    Exemplo: selecionar_ids_arquivo('dataset/produtos_id_titulo.csv', 'dataset/ids_teste')
    """
    dict_id_selecionados = {}
    with open(files_ids_selecionados, newline='') as csvfile_reader_ids_selecionados:
        with open(file_ids_total, newline='') as csvfile_reader_total:
            with open('arquivo_reduzido.csv', 'w', newline='') as csvfile_write_saida:
                ids_selecionados = csv.reader(csvfile_reader_ids_selecionados)
                for row in ids_selecionados:
                    dict_id_selecionados.__setitem__(row[0], True)

                saida_reduzida = csv.writer(csvfile_write_saida, quotechar='|')  # O | permite caracter "
                ids_all = csv.reader(csvfile_reader_total)
                for row in ids_all:
                    if (dict_id_selecionados.__contains__(row[0])):
                        text = "\"" + str(row[1]) + "\""
                        saida_reduzida.writerow([row[0], text])


def arquivo_para_corpus(file_corpus, id_column):
    """ Transforma o arquivo .csv no formato 'list' compatível para o tfidf."""
    corpus = []
    with open(file_corpus, 'r', newline='') as out:
        csv_reader = csv.reader(out, delimiter=',')
        # next(csv_reader)  # Pular cabecalho
        for row in csv_reader:
            corpus.append(row[id_column])
    return corpus

def arquivo_para_corpus_space(file_corpus):
    """ Transforma o arquivo .csv no formato 'list' compatível para o tfidf."""
    corpus = []
    with open(file_corpus, 'r', newline='') as out:
        csv_reader = csv.reader(out, delimiter=' ')
        # next(csv_reader)  # Pular cabecalho
        for row in csv_reader:
            corpus.append(row)
    return corpus


def frequencia_termos(listas):
    """ Retorna um dict ordenado pela freqência dos termos da lista."""
    terms_unicos = dict()
    for lista in listas:
        for termo in lista:
            if (terms_unicos.__contains__(termo) == False):
                terms_unicos.__setitem__(termo, 1)
            else:
                terms_unicos.__setitem__(termo, terms_unicos.get(termo) + 1)
    # Ordena pela frequencia, frequencias iguais utiliza ordem alfabetica
    terms_unicos = sorted(terms_unicos.items(), key=lambda e: (-e[1], e[0]))
    return terms_unicos


def soma_das_frequencia(listas):
    """ Soma as frequencias dos termos de uma lista de listas. """
    termos_unicos = {}
    for lista in listas:
        for termo_freq in lista:
            if termos_unicos.__contains__(termo_freq[0]) == True:
                termos_unicos.__setitem__(termo_freq[0], termo_freq[1] + termos_unicos.__getitem__(
                    termo_freq[0]))  # Valor atual + anterior
            else:
                termos_unicos.__setitem__(termo_freq[0], termo_freq[1])
    return sorted(termos_unicos.items(), key=lambda e: (-e[1], e[0]))


def soma_das_frequencia2(listas):
    """ Soma as frequencias dos termos de uma lista de listas. """
    termos_unicos = {}
    for termo_freq in listas:
        if termos_unicos.__contains__(termo_freq[0]) == True:
            termos_unicos.__setitem__(termo_freq[0], termo_freq[1] + termos_unicos.__getitem__(
                termo_freq[0]))  # Valor atual + anterior
        else:
            termos_unicos.__setitem__(termo_freq[0], termo_freq[1])
    return sorted(termos_unicos.items(), key=lambda e: (-e[1], e[0]))


def coloca_espaco_vetor(lista):
    """ Coloca espaço entre os elementos da lista."""
    txt = ""
    for i in range(lista.__len__()):
        if i == lista.__len__() - 1:
            txt = txt + lista[i]
        else:
            txt = txt + lista[i] + " "
    return txt


def indice_maior_element_numpy(vet_numpy):
    """ Return the index of max value element in array numpy. """

    result = numpy.where(vet_numpy == numpy.amax(vet_numpy))
    # print('List of Indices of maximum element :', result[1])
    # print(str(result[1]).replace("[", "").replace("]",""))
    return str(result[1]).replace("[", "").replace("]", "")


def save_model_lda(corpus, file_lda):
    """ Salvar modelo LDA em arquivo."""

    vectorizer = TfidfVectorizer()  # Utilizar o metodo tfidf
    lda = LatentDirichletAllocation(n_components=5, random_state=0)  # Utilizar 5 topicos
    ini = timeit.default_timer()
    X = vectorizer.fit_transform(corpus)  # Transformar texto em matrix
    # X_TF = vectorizer.fit_transform(corpus).tocsr()
    # X = sp.csr_matrix( ( np.ones(len(X_TF.data)), X_TF.nonzero() ), shape=X_TF.shape )
    lda.fit(X)  # Treinar com o texto
    print("Train LDA: %f" % (timeit.default_timer() - ini))

    # Save all data necessary for later prediction
    dic = vectorizer.get_feature_names()  # nome das features
    model = (dic, lda.components_, lda.exp_dirichlet_component_, lda.doc_topic_prior_)
    with open(file_lda, 'wb') as fp:
        pickle.dump(model, fp)


def k_elements_list(lista, k):
    """ Retorna uma lista com k primeiros elementos."""
    new_lista = []
    i = 1
    for e in lista:
        new_lista.append(e)
        if (i == k):
            break
        i += 1
    return new_lista


def save_one_column_csv(file_csv, id_column):
    """ Save one column csv. """
    with open(name_out(file_csv), 'w', newline='') as csv_write:
        rows_out = csv.writer(csv_write, quotechar='|')  # O | permite caracter "
        with open(file_csv, newline='') as csv_reader:
            rows = csv.reader(csv_reader)
            for row in rows:
                rows_out.writerow([row[id_column - 1]])


def name_out(file_csv):
    """ Return name of out file."""
    name = os.path.basename(file_csv)
    file_name = os.path.splitext(name)[0]
    file_type = os.path.splitext(name)[1]
    file_location = os.path.dirname(file_csv) + "/"
    date = "_" + datetime.now().strftime('%d-%m-%Y.%H-%M-%S')
    return file_location + file_name + date + file_type


def no_repeat_id(file_csv):
    """ No repeat id in file csv."""
    file_out = name_out(file_csv)
    dict_row = {}

    with open(file_out, 'w', newline='') as csv_write:
        with open(file_csv, newline='') as csv_reader:
            rows = csv.reader(csv_reader, quotechar='|')
            for row in rows:
                dict_row.__setitem__(row[0], row)
            rows_out = csv.writer(csv_write, quotechar='|')  # O | permit character "
            for id, row in dict_row.items():
                rows_out.writerow(row)


def add_caracter_column(file_csv):
    # add_caracter_column("experiment/tck_resultado_com_seed.csv")
    with open('experiment/tags.csv', 'w', newline='') as csv_write:

        csv.register_dialect('myDialect',
                             delimiter='|',
                             quoting=csv.QUOTE_NONE,
                             skipinitialspace=True)

        rows_out = csv.writer(csv_write, dialect='myDialect')  # O | permite caracter "
        with open(file_csv, newline='') as csv_reader:
            rows = csv.reader(csv_reader)
            for row in rows:
                list_k = row[0].split(" ")
                s = ""

                i=1
                for k in list_k:
                    s = s + " " + k
                    if i == 3:
                        break
                    i+=1
                row[0] = row[0].replace(" ", ",")

                if len(list_k) > 4:
                    one = list_k[0] +" " + list_k[1]
                    dois = list_k[0] +" " + list_k[2]
                    tres = list_k[0] +" " + list_k[3]
                    quatro = list_k[1] +" " + list_k[2]
                    final = list_k[0] +"," + list_k[1] +"," +list_k[2] +"," +list_k[3] +"," +list_k[4] +"," +one +"," +dois +"," +tres +"," + quatro +"," +row[0] +"," +s
                else:
                    final = row[0] +"," + s

                rows_out.writerow([final])


def k_max_index(list, k):
    """ Return index of max values.
    Example:
    r = [0.5, 0.7, 0.3, 0.3, 0.3, 0.4, 0.5]
    print(k_max_index(r, 3))
    """''
    list_m = list.copy()
    max_index = []
    k_max = nlargest(k, list_m)
    for i in k_max:
        max_index.append(list_m.index(i))
        list_m[list_m.index(i)] = -1
    return max_index


def amount_terms_corpus (X, vec_terms):
    """ Amount of terms in corpus """
    terms_total = {}
    for i in range(X.shape[1]):
        terms_total[vec_terms[i]] = sum(X.getcol(i).toarray())[0]
    return terms_total

def save_dict_file(file, dict):
    """Save dict in file"""
    with open(file, 'w', newline='') as csv_write:
        json.dump(dict, csv_write)

def load_dict_file(file):
    """Load dict in file"""
    with open(file, 'r', newline='') as csv_reader:
        return json.load(csv_reader)

def count_same_in_index(list,list2):
    """ Return the amount of value in index."""
    count=0
    for i in range(len(list)):
        if list[i] != 0 and list2[i] != 0:
            count+=1
    return count

def list_files(dir):
    files_name =[]
    for r, d, files_array in os.walk(dir):
        for f in files_array:
            files_name.append(f)
    return files_name

r = [0, 7, 0, 3, 0, 1, 0]
r2 = [0, 7, 0, 3, 0, 2, 1]


import csv
import shutil  # Copy files


file = "../stage_1_train.csv"
folder_train = "//home/claudiovaliense/kaggle/rsna/stage_1_train_images/"
dst = "//home/claudiovaliense/kaggle/rsna/normal"
types = dict()
types['epidural'] = 0
types['intraparenchymal'] = 0
types['intraventricular'] = 0
types['subarachnoid'] = 0
types['subdural'] = 0
types['any'] = 0
#types['normal'] = 0


def return_id_label():
    """ Amount of subtypes"""
    with open(file, 'r', newline='') as csv_reader:
        rows = csv.reader(csv_reader, quotechar=',')
        next(rows)

        id_types=dict()
        cont=0
        types = dict()
        for row in rows:
            if str(row[0]).__contains__('epidural') and row[1] != '0':
                types['epidural'] = 1
            if str(row[0]).__contains__('intraparenchymal') and row[1] != '0':
                types['intraparenchymal'] = 1
            if str(row[0]).__contains__('intraventricular') and row[1] != '0':
                types['intraventricular'] = 1
            if str(row[0]).__contains__('subarachnoid') and row[1] != '0':
                types['subarachnoid'] = 1
            if str(row[0]).__contains__('subdural') and row[1] != '0':
                types['subdural'] = 1
            if str(row[0]).__contains__('any') and row[1] != '0':
                types['any'] = 1
            #if str(row[0]).__contains__('any') and row[1] == '0':
             #   types['normal'] = 1

            cont+=1
            if cont == 6:
                cont = 0
                id = row[0].split('_')[0] + '_' + row[0].split('_')[1] + ".dcm"
                id_types[id] = types # atribui o id e seus types
                types = dict()
                types['epidural'] = 0
                types['intraparenchymal'] = 0
                types['intraventricular'] = 0
                types['subarachnoid'] = 0
                types['subdural'] = 0
                types['any'] = 0
                #types['normal'] = 0

        return id_types

def amount_types():
    """ Amount of subtypes"""
    with open(file, 'r', newline='') as csv_reader:
        rows = csv.reader(csv_reader, quotechar=',')
        next(rows)
        epidural = []
        intraparenchymal =[]
        intraventricular=[]
        subarachnoid=[]
        subdural=[]
        any=[]
        normal=[]
        for row in rows:
            if str(row[0]).__contains__('epidural') and row[1] != '0':
                epidural.append(row[0])
            if str(row[0]).__contains__('intraparenchymal') and row[1] != '0':
                intraparenchymal.append(row[0])
            if str(row[0]).__contains__('intraventricular') and row[1] != '0':
                intraventricular.append(row[0])
            if str(row[0]).__contains__('subarachnoid') and row[1] != '0':
                subarachnoid.append(row[0])
            if str(row[0]).__contains__('subdural') and row[1] != '0':
                subdural.append(row[0])
            if str(row[0]).__contains__('any') and row[1] != '0':
                any.append(row[0])
            if str(row[0]).__contains__('any') and row[1] == '0':
                normal.append(row[0])

        types['epidural'] = epidural
        types['intraparenchymal'] = intraparenchymal
        types['intraventricular'] = intraventricular
        types['subarachnoid'] = subarachnoid
        types['subdural'] =subdural
        types['any'] = any
        types['normal'] = normal
        return types

def ids_type(types):
    """Copy file of epidural"""
    print(types)
    for id in types['epidural']:
        id = id.split('_')[0] +'_' + id.split('_')[1] +".dcm"
        shutil.copy(folder_train+id, dst)

def ids_type_normal(types):
    """Copy file of epidural"""
    count=0
    for id in types['normal']:
        id = id.split('_')[0] +'_' + id.split('_')[1] +".dcm"
        shutil.copy(folder_train+id, dst)
        if count == 2761:
            break
        count+=1


return_id_label()
#ids_type_normal(amount_types())
#types = amount_types()
#print(len(types['epidural'])):

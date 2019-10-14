import lib
import os

input_path = "dataset/epidural"
input_folder = os.fsencode(input_path)
files = os.listdir(input_folder)
files.sort()

slices = []

for file in files:
    # if os.fsdecode(file) == "ID_3580adc72.dcm": # "ID_635f084fc.dcm": # "ID_559b1d8f7.dcm": #"ID_894a589ad.dcm":
    if os.fsdecode(file) == "ID_0ed10ec08.dcm":
        file = os.fsdecode(file)
        filename = "{}/{}".format(input_path, file)
        image = lib.read_image(filename)
        lib.histogram(image, True)
        lib.plot("original: {} ".format(file), image)
        # features
        hematoma = lib.substance_interval(image, 40, 90)
        white_matter = lib.substance_interval(image, 20, 30)
        blood = lib.substance_interval(image, 30, 45)
        bone = lib.substance_interval(image, 600, 4000)

        #lib.plot("blood: {}".format(file), blood)
        #lib.plot("hematoma: {}".format(file), hematoma)

        lib.histogram(blood, True)
print("Done")

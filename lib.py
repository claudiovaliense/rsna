from matplotlib import pyplot as plt
import numpy as np
import pydicom


def read_image(filename):
    ds = pydicom.dcmread(filename)
    b = ds.RescaleIntercept
    m = ds.RescaleSlope
    # hu = pixel_value * slope + intercept
    image = m * ds.pixel_array + b
    return image


def plot(title, image):
    plt.imshow(image, cmap=plt.cm.bone)
    plt.title(title)
    plt.show()

def substance_interval(image, min, max):
    "Return image of substante"
    # https://www.sciencedirect.com/topics/medicine-and-dentistry/hounsfield-scale
    # https://books.google.com.br/books?id=R46dDwAAQBAJ&pg=PA744&lpg=PA744&dq=dicom+bone+interval+hu&source=bl&ots=0rystnK6I5&sig=ACfU3U2DWVt2BEnAVenPIxs45QKklbV69g&hl=pt-BR&sa=X&ved=2ahUKEwjQ37nsgpvlAhXnmuAKHdn2BQsQ6AEwDnoECAoQAQ#v=onepage&q=dicom%20bone%20interval%20hu&f=false
    # http://radclass.mudr.org/content/hounsfield-units-scale-hu-ct-numbers
    img = image.copy()
    img[img > max] = 0
    img[img < min] = 0
    return img


def histogram(image, remove_min=False):
    max = np.max(image)
    min = np.min(image)
    if remove_min:
        min = np.min(image[image > min])
    plt.hist(image.ravel(), 256, [min, max])
    plt.title("histogram")
    plt.show()

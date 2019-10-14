from skimage import io
import matplotlib.pyplot as plt
from skimage import filters
from skimage import  data

#file_name = 'vermelho.png'
file_name = '../grayscale_chart.jpg'

image = io.imread(file_name)

plt.figure()
plt.title('minha imagem')
print('dimensão imagem: ', image.ndim)

if image.ndim == 2:
    plt.imshow(image, cmap=plt.cm.gray)
else:
    plt.imshow(image)
plt.show()

coins = data.coins()
plt.imshow(coins)
print(coins)

threshold_value = filters.threshold_otsu(coins)

print(threshold_value)
plt.show()

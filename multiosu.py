from skimage.color import label2rgb
from skimage import data
import skimage
image = data.camera()
thresholds = threshold_multiotsu(image)
regions = np.digitize(image, bins=thresholds)
regions_colorized = label2rgb(regions)

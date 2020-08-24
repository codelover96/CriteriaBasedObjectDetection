import gdal
import ogr
from skimage import exposure
from skimage.segmentation import quickshift, slic
import geopandas as gpd
import numpy as np
import time

filename = ''
file_extension = '.tif'
naip_fn = filename + file_extension

driverTiff = gdal.GetDriverByName('GTiff')
naip_ds = gdal.Open(naip_fn)
nbands = naip_ds.RasterCount
band_data = []

for i in range(1, nbands + 1):
    band = naip_ds.GetRasterBand(i).ReadAsArray()
    band_data.append(band)
band_data = np.dstack(band_data)
img = exposure.rescale_intensity(band_data)
# print("number of bands in : "+naip_fn+" is ", nbands)



## do segmentation, different options with quickshift and slic (only use one of the next two lines)
#segments = quickshift(img, ratio=0.99, max_dist=5, convert2lab=False)
## segments = slic(img, n_segments=500000, compactness=0.1)
#print('segments complete', time.time())

# save segments to raster
#segments_fn = 'seg_slic_'+filename+'.tif'
#segments_ds = driverTiff.Create(segments_fn, naip_ds.RasterXSize, naip_ds.RasterYSize,1, gdal.GDT_Float32)
#segments_ds.SetGeoTransform(naip_ds.GetGeoTransform())
#segments_ds.SetProjection(naip_ds.GetProjectionRef())
#segments_ds.GetRasterBand(1).WriteArray(segments)
#segments_ds = None
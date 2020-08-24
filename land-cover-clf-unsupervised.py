from sklearn.cluster import KMeans
import gdal
import numpy as np
import time

# read in image to classify with gdal
filename = ''
file_extension = '.tif'
naip_fn = filename + file_extension
driverTiff = gdal.GetDriverByName('GTiff')
naip_ds = gdal.Open(naip_fn)
nbands = naip_ds.RasterCount

# create an empty array, each column of the empty array will hold one band of data from the image
# loop through each band in the image nad add to the data array
data = np.empty((naip_ds.RasterXSize*naip_ds.RasterYSize, nbands))
for i in range(1, nbands+1):
    band = naip_ds.GetRasterBand(i).ReadAsArray()
    data[:, i-1] = band.flatten()

# set up the k-means classification, fit, and predict
kmeansclusters = 11
km = KMeans(kmeansclusters)
km.fit(data)
km.predict(data)

# format the predicted classes to the shape of the original image
out_dat = km.labels_.reshape((naip_ds.RasterYSize, naip_ds.RasterXSize))

# save the original image with gdal
# clfds = driverTiff.Create('/classified/'+naip_fn+'-classified'+str(kmeansclusters)+time.asctime()+'.tif', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_Float32)
clfds = driverTiff.Create(filename+'-'+str(kmeansclusters)+'clusters'+'.tif', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_Float32)
clfds.SetGeoTransform(naip_ds.GetGeoTransform())
clfds.SetProjection(naip_ds.GetProjection())
clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
clfds.GetRasterBand(1).WriteArray(out_dat)
clfds = None

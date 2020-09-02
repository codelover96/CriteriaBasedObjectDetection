"""
Convert from rle aka Run-Length encoding to bounding boxes and then to YOLO format
"""
import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import label
from pathlib import Path


output_dir = Path("G:/Programming/Python/Edge Detection/bboxes")
ship_dir = Path("G:/airbus-ship-detection")
train_image_dir = os.path.join(ship_dir, 'train_v2')
test_image_dir = os.path.join(ship_dir, 'test_v2')
seg_csv = os.path.join(ship_dir, 'train_ship_segmentations_v2.csv')
seg_csv_only_ships = os.path.join(ship_dir, 'train_ship_segmentations_v2_only_ship_images.csv')
bboxes_dir = os.path.join(ship_dir, 'bboxes')
txt_bboxes_dir = os.path.join(ship_dir, "txt")


def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list, all_masks=None):
    # Take the individual ship masks and create a single mask array for all ships
    if all_masks is None:
        all_masks = np.zeros((768, 768), dtype=np.int16)
    # if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


masks = pd.read_csv(seg_csv_only_ships)
print(masks.shape[0], 'masks found')
print(masks['ImageId'].value_counts().shape[0])


images_with_ship = masks.ImageId[masks.EncodedPixels.isnull() == False]
images_with_ship = np.unique(images_with_ship.values)
print('There are ' + str(len(images_with_ship)) + ' image files with masks')
print("====================================")
for i in range(0, len(images_with_ship)):
    image = images_with_ship[i]
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    img_0 = cv2.imread(train_image_dir + '/' + image)
    rle_0 = masks.query('ImageId=="' + image + '"')['EncodedPixels']
    mask_0 = masks_as_image(rle_0)
    lbl_0 = label(mask_0)
    props = regionprops(lbl_0)
    img_1 = img_0.copy()
    image_name = image
    print('Image name :', image_name, ' has ', len(props), ' ships')

    print('---------------------------------------------------')
    # create file to write to
    # also remove file extension from variable image_name and add .txt extension
    file = open(os.path.join(txt_bboxes_dir, image_name.rsplit(".", 1)[0] + ".txt"), "w+")

    for prop in props:
        cv2.rectangle(img_1, (prop.bbox[1], prop.bbox[3]), (prop.bbox[4], prop.bbox[0]), (0, 0, 255), 2)
        # draw circles of bottom left and top right points.
        image = cv2.circle(img_1, (prop.bbox[1], prop.bbox[3]), radius=2, color=(0, 0, 255), thickness=-1)
        image = cv2.circle(img_1, (prop.bbox[4], prop.bbox[0]), radius=2, color=(0, 255, 0), thickness=-1)
        # calculate middle point of rectangular (basic geometry)
        mid_x = (prop.bbox[1] + prop.bbox[4]) / 2
        mid_y = (prop.bbox[3] + prop.bbox[0]) / 2
        # draw a circle at middle point
        image = cv2.circle(img_1, (int(mid_x), int(mid_y)), radius=2, color=(255, 255, 0), thickness=-1)
        # yolo needs width, height and middle point to be normalized to image dimensions.
        # airbus ship detection dataset images are, 768x768
        width = abs(prop.bbox[1] - prop.bbox[4]) / 768  # normalized to image size
        height = abs(prop.bbox[0] - prop.bbox[3]) / 768  # same
        # normalize and round to 6 decimal points
        # for more info on yolo format : https://github.com/AlexeyAB/Yolo_mark/issues/60
        line = "0" + " " + str(round(mid_x / 768, 6)) + " " + str(round(mid_y / 768, 6)) + " " + str(
            round(width, 6)) + " " + str(round(height, 6))
        print(line)
        file.write(line + "\n")
        # image = cv2.circle(img_1, (int(mid_x), int(mid_y)), radius=3, color=(0, 255, 0), thickness=-1)
        # print(mid_x)
        # print(mid_y)
        # print('Rectangle prop.bbox[0]: '+str(prop.bbox[0]))
        # print('Rectangle prop.bbox[1]: '+str(prop.bbox[1]))
        # print('Rectangle prop.bbox[3]: '+str(prop.bbox[3]))
        # print('Rectangle prop.bbox[4]: '+str(prop.bbox[4]))

    file.close()
    print('---------------------------------------------------')
    print("\n")
    # ax1.imshow(img_0)
    # ax1.set_title(image)
    # ax2.set_title('Mask')
    # ax3.set_title('Image with derived bounding box')
    # ax2.imshow(mask_0[..., 0], cmap='gray')

    # plt.imshow(img_1)

    # ax3.annotate('a', xy=(344, 501), xytext=(10, 4), arrowprops=dict(facecolor='black', shrink=0))
    # ax3.annotate('b', xy=(448, 465), xytext=(10, 4), arrowprops=dict(facecolor='red', shrink=0))
    # ax3.annotate('b', xy=(396, 483), xytext=(10, 4), arrowprops=dict(facecolor='red', shrink=0))

    # image_rgb = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(os.path.join(bboxes_dir, image_name), img_1)
    # plt.text(344, 501, '344,501')
    # plt.text(448, 465, '448,465')
    # centerbbox_x1 =
    # centerbbox_y1 =

    # plt.show()

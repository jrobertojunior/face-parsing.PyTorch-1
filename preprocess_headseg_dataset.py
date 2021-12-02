import cv2 as cv
import numpy as np
import os
from xml.dom.minidom import parse


def preprocess(labels_path, output_path):
    # check if the directory exists. if not, create it
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    count = 0
    for filename in os.listdir(labels_path):
        # only works with .png files
        if not filename.endswith(".png"):
            continue

        label_path = os.path.join(labels_path, filename)

        mask = segment_labels(label_path)

        # save the mask
        mask_path = os.path.join(output_path, filename[:-4] + ".png")
        print(f"{count}. saving {mask_path}", end='\r')
        cv.imwrite(mask_path, mask)

        count += 1


def segment_labels(label_path):
    atts = {
        "background": (0, 0, 0),
        "mouth": (255, 0, 0),
        "eyes": (0, 255, 0),
        "nose": (0, 0, 255),
        "face": (128, 128, 128),
        "hair": (255, 255, 0),
        "eyebrows": (255, 0, 255),
        "ears": (0, 255, 255),
        "teeth": (255, 255, 255),
        "beard": (255, 192, 192),
        "sunglasses": (0, 128, 128),
    }

    label = cv.imread(label_path)
    mask = np.zeros(label.shape, dtype=np.uint8)

    for i, att in enumerate(atts, 1):
        # segment mask
        seg_mask = cv.inRange(label, atts[att], atts[att])

        # write segmented mask in grayscale to mask
        mask[seg_mask == 255] = i

    return mask


def reorganize_data(path, dest):
    """
    Reorganizes mut1ny's face/head dataset from folder schema:
        headsegmentation_dataset_ccncsa/
            female03_blackhair_browneyes_env08/
            [...]
            labels/
            [...]

    to folder schema:
        dataset/
            images/
            labels/
    """
    # check if the directory exists. if not, create it
    if not os.path.exists(dest):
        os.mkdir(dest)
        os.mkdir(os.path.join(dest, "images"))
        os.mkdir(os.path.join(dest, "labels"))

    print("parsing traning.xml...")
    document = parse(path + "/training.xml")

    # list of all tags for images
    images = document.getElementsByTagName("srcimg")
    # list of all tags for labels
    labels = document.getElementsByTagName("labelimg")

    assert len(images) == len(labels)

    for i in range(len(images)):
        image_path = os.path.join(path, images[i].getAttribute("name"))
        label_path = os.path.join(path, labels[i].getAttribute("name"))

        # copy image to dest/images
        out_image_path = os.path.join(dest, "images", str(i) + ".png")
        cv.imwrite(out_image_path, cv.imread(image_path))
        # print(f"{i}. copied {image_path} to {out_image_path}")

        # copy label to dest/labels
        out_labels_path = os.path.join(dest, "labels", str(i) + ".png")
        cv.imwrite(out_labels_path, cv.imread(label_path))
        # print(f"{i}. copied {label_path} to {out_labels_path}")

        print(f"{i}. saved {out_image_path} and {out_labels_path}", end='\r')

    


# reorganize_data("./headsegmentation_dataset_ccncsa/", "./dataset")
preprocess("dataset/labels", "dataset/mask")

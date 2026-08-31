import gzip
import struct

import numpy as np


def load_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        header: bytes = file.read(16)
        magic, image_count, rows, columns = struct.unpack(">IIII", header)

        if magic != 2051:
            raise ValueError("invalid MNIST image file")

        raw_pixels: bytes = file.read()
        pixels = np.frombuffer(raw_pixels, dtype=np.uint8)

    images = pixels.reshape(image_count, rows * columns)
    images = images.astype(np.float64) / 255.0
    return images


def load_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        header: bytes = file.read(8)
        magic, label_count = struct.unpack(">II", header)

        if magic != 2049:
            raise ValueError("invalid MNIST label file")

        raw_labels: bytes = file.read()
        labels = np.frombuffer(raw_labels, dtype=np.uint8)

    if len(labels) != label_count:
        raise ValueError("label count does not match file data")

    return labels.astype(np.int64)


class Dataset:
    def __init__(self, image_path: str, label_path: str):
        self.images = load_images(image_path)
        self.labels = load_labels(label_path)

        if len(self.images) != len(self.labels):
            raise ValueError("image and label counts do not match")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = int(self.labels[index])
        return image, label

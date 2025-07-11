import os
import glob
import numpy as np
import rasterio
import tensorflow as tf

# Configuration constants for tile size and batch
TILE_SIZE = 256
BATCH_SIZE = 8


def list_tile_paths(data_dir, timestamps):
    """
    Return a list of paired tile paths across timestamps.
    """
    pairs = []
    base = os.path.abspath(data_dir)
    for t in timestamps:
        tdir = os.path.join(base, t)
        imgs = sorted(glob.glob(os.path.join(tdir, "*.tif")))
        pairs.append(imgs)
    return list(zip(*pairs))


def read_pair(path1, path2):
    """
    Read two GeoTIFF files, scale reflectance, and return HWC arrays.
    """
    with rasterio.open(path1) as src1, rasterio.open(path2) as src2:
        arr1 = src1.read([1,2,3]) / 10000.0
        arr2 = src2.read([1,2,3]) / 10000.0
    arr1 = np.transpose(arr1, (1,2,0))
    arr2 = np.transpose(arr2, (1,2,0))
    return arr1.astype(np.float32), arr2.astype(np.float32)


def make_dataset(pairs):
    """
    Build a tf.data.Dataset yielding batches of (img1, img2) pairs.
    """
    def gen():
        for p1, p2 in pairs:
            x, y = read_pair(p1, p2)
            yield x, y

    ds = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=((TILE_SIZE, TILE_SIZE, 3), (TILE_SIZE, TILE_SIZE, 3))
    )
    ds = ds.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

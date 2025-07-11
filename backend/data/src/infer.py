import numpy as np
import rasterio
from model import ChangeDetector
from data_utils import read_pair


def detect_change(model_path, img1_path, img2_path, out_path):
    # Load model
    model = ChangeDetector()
    model.load_weights(model_path)

    # Read and predict
    x1, x2 = read_pair(img1_path, img2_path)
    inp1 = np.expand_dims(x1, 0)
    inp2 = np.expand_dims(x2, 0)
    pred = model.predict([inp1, inp2])[0, ..., 0]

    # Save binary mask
    with rasterio.open(img1_path) as src:
        profile = src.profile
        profile.update(count=1, dtype=rasterio.uint8)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write((pred > 0.5).astype(rasterio.uint8), 1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Detect changes between two tiles')
    parser.add_argument('--model', required=True)
    parser.add_argument('--img1', required=True)
    parser.add_argument('--img2', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    detect_change(args.model, args.img1, args.img2, args.out)
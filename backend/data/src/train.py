import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from data_utils import list_tile_paths, make_dataset
from model import ChangeDetector

# Project-specific config
DATA_DIR = "data/tiles"
TIMESTAMPS = ["2021-06", "2022-06"]
EPOCHS = 20

if __name__ == "__main__":
    # Prepare data
    pairs = list_tile_paths(DATA_DIR, TIMESTAMPS)
    dataset = make_dataset(pairs)

    # Build and compile model
    model = ChangeDetector()
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks for checkpointing and TensorBoard
    os.makedirs('models', exist_ok=True)
    ckpt = ModelCheckpoint('models/best_model.h5', save_best_only=True)
    tb = TensorBoard(log_dir='logs/fit')

    # Train
    model.fit(
        dataset,
        epochs=EPOCHS,
        callbacks=[ckpt, tb]
    )

    # Save final model
    model.save('models/final_model.h5')
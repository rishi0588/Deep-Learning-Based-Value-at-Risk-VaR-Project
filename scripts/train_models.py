import numpy as np
import os
from sklearn.metrics import mean_squared_error
from scripts.models import build_mlp, build_cnn, build_lstm, build_transformer
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Folder paths
DATA_FOLDER = "data/processed"
RESULTS_FOLDER = "results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

EPOCHS = 40
BATCH_SIZE = 64

callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

TEST_SPLIT = 0.8

# Model builders dictionary
MODEL_BUILDERS = {
    "MLP": build_mlp,
    "CNN1D": build_cnn,
    "LSTM": build_lstm,
    "Transformer": build_transformer
}

# Loop through each processed dataset
for file in os.listdir(DATA_FOLDER):
    if not file.endswith(".npz"):
        continue

    stock_name = file.replace("_seq.npz", "")
    print(f"\n📈 Training models for {stock_name}...")

    # Load data
    data = np.load(os.path.join(DATA_FOLDER, file))
    X, y = data["X"], data["y"]

    # Split into train/test
    split_idx = int(TEST_SPLIT * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    input_shape = (X_train.shape[1], X_train.shape[2])

    for model_name, builder in MODEL_BUILDERS.items():
        print(f"\n🚀 Training {model_name} model...")
        model = builder(input_shape)

        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        # Save results
        np.savez_compressed(f"{RESULTS_FOLDER}/{stock_name}_{model_name}_preds.npz",
                            y_test=y_test, preds=preds)
        with open(f"{RESULTS_FOLDER}/{stock_name}_{model_name}_mse.txt", "w") as f:
            f.write(f"MSE: {mse}\n")

        print(f"✅ {model_name} done. MSE = {mse:.6f}")

print("\n🎯 All models trained and predictions saved!")

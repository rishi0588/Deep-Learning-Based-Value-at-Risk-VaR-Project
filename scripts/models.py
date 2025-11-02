from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization,
    LSTM, Bidirectional, Input, MultiHeadAttention, Add,
    LayerNormalization, GlobalAveragePooling1D
)
import tensorflow as tf


# --- 1️⃣ MLP (Deeper, regularized) ---
def build_mlp(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model


# --- 2️⃣ CNN1D (Residual-style 1D CNN) ---
def build_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model


# --- 3️⃣ LSTM (Stacked BiLSTM with regularization) ---
def build_lstm(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss='mse')
    return model


# --- 4️⃣ Transformer (Modernized with feedforward block + residuals) ---
def build_transformer(input_shape, num_heads=4, ff_dim=64):
    inputs = Input(shape=input_shape)

    # Multi-Head Self-Attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(inputs, inputs)
    attn_output = Dropout(0.2)(attn_output)
    out1 = Add()([inputs, attn_output])
    out1 = LayerNormalization()(out1)

    # Feed-Forward Network
    ffn = Dense(ff_dim * 2, activation='relu')(out1)
    ffn = Dense(ff_dim, activation='relu')(ffn)
    ffn = Dropout(0.2)(ffn)
    out2 = Add()([out1, ffn])
    out2 = LayerNormalization()(out2)

    # Global pooling + dense projection
    x = GlobalAveragePooling1D()(out2)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), loss='mse')
    return model

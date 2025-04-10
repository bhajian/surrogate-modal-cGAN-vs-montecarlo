# train_cgan.py — Optimized cGAN Training with Full GPU Utilization

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm, trange
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
import joblib
import time

# === Mixed Precision ===
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# === Config ===
SEQ_IN = 250
SEQ_OUT_LIST = [1, 30, 90, 250]
z_dim = 100
batch_size = 1024  # Maximize batch size for GPU
lr = 1e-4

# === Data Loading ===
data_path = 'stock_data/eod'
tickers = pd.read_csv('nasdaq_tickers.csv')['ticker'].tolist()

X_cond, Y_target = [], []

for ticker in tqdm(tickers, desc="Slicing each stock"):
    filepath = os.path.join(data_path, f"{ticker}.csv")
    if not os.path.isfile(filepath):
        continue

    df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
    if 'close' not in df.columns or df['close'].isna().sum() > 0:
        continue

    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna().values

    for horizon in SEQ_OUT_LIST:
        for i in range(len(log_returns) - (SEQ_IN + horizon)):
            cond = log_returns[i:i+SEQ_IN]
            future = log_returns[i+SEQ_IN:i+SEQ_IN+horizon]
            padded_future = np.pad(future, (0, max(SEQ_OUT_LIST) - horizon))
            X_cond.append(cond)
            Y_target.append(padded_future)

X_cond = np.array(X_cond, dtype=np.float32)
Y_target = np.array(Y_target, dtype=np.float32)

if X_cond.shape[0] == 0:
    raise ValueError("❌ No training samples generated! Check SEQ_IN, data size.")

print(f"✅ Training samples: {X_cond.shape[0]}")

# === Normalize ===
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_cond_scaled = scaler_X.fit_transform(X_cond).astype(np.float32)
Y_target_scaled = scaler_Y.fit_transform(Y_target).astype(np.float32)

# === Build Models ===
def build_generator(z_dim, cond_dim, output_dim):
    inputs = layers.Input(shape=(z_dim + cond_dim,))
    x = layers.Dense(2048, activation='relu')(inputs)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(output_dim, dtype='float32')(x)  # Ensure float32 output for compatibility
    return tf.keras.Model(inputs, x)

def build_discriminator(cond_dim, input_dim):
    inputs = layers.Input(shape=(cond_dim + input_dim,))
    x = layers.Dense(2048)(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return tf.keras.Model(inputs, x)

G = build_generator(z_dim, SEQ_IN, max(SEQ_OUT_LIST))
D = build_discriminator(SEQ_IN, max(SEQ_OUT_LIST))

# === Optimizers ===
bce = tf.keras.losses.BinaryCrossentropy()
g_opt = tf.keras.optimizers.Adam(learning_rate=lr)
d_opt = tf.keras.optimizers.Adam(learning_rate=lr)

# === Dataset ===
dataset = (
    tf.data.Dataset.from_tensor_slices((X_cond_scaled, Y_target_scaled))
    .shuffle(buffer_size=50000)
    .batch(batch_size, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

@tf.function
def train_step(condition, real_output):
    batch_size = tf.shape(condition)[0]
    noise = tf.random.normal((batch_size, z_dim))
    input_gen = tf.concat([noise, condition], axis=1)

    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_output = G(input_gen, training=True)
        real_input = tf.concat([condition, real_output], axis=1)
        fake_input = tf.concat([condition, fake_output], axis=1)

        d_real = D(real_input, training=True)
        d_fake = D(fake_input, training=True)

        d_loss = bce(tf.ones_like(d_real), d_real) + bce(tf.zeros_like(d_fake), d_fake)
        g_loss = bce(tf.ones_like(d_fake), d_fake)

    d_grads = d_tape.gradient(d_loss, D.trainable_variables)
    g_grads = g_tape.gradient(g_loss, G.trainable_variables)

    d_opt.apply_gradients(zip(d_grads, D.trainable_variables))
    g_opt.apply_gradients(zip(g_grads, G.trainable_variables))
    return d_loss, g_loss

# === Train ===
epochs = 3000
print("🚀 Training...")
for epoch in trange(epochs):
    for cond_batch, real_batch in dataset:
        d_loss, g_loss = train_step(cond_batch, real_batch)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: D = {d_loss.numpy():.4f} | G = {g_loss.numpy():.4f}")

# === Save Artifacts ===
os.makedirs("models", exist_ok=True)
G.save('models/generator_stockwise_cgan.h5')
D.save('models/discriminator_stockwise_cgan.h5')
joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_Y, 'models/scaler_Y.pkl')
print("✅ Models and scalers saved.")

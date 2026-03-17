import mlflow
import mlflow.pytorch
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os


mlflow.set_experiment("Assignment3_Salma")

CSV_PATH = "fashion-mnist_train.csv"
BATCH_SIZE = 64
NOISE_DIM = 50
EPOCHS = 5
IMG_SIZE = 28
SAVE_DIR = "generated_faces"

os.makedirs(SAVE_DIR, exist_ok=True)

# Load CSV Dataset (Fashion-MNIST)
df = pd.read_csv(CSV_PATH)

# Drop label column
x_train = df.drop("label", axis=1).values

# Reshape to images
x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32")

# Normalize to [-1,1]
x_train = (x_train - 127.5) / 127.5

dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(60000).batch(BATCH_SIZE)

# Generator
def build_generator():
    model = tf.keras.Sequential()

    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7,7,256)))

    model.add(layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding="same",use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding="same",use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1,(5,5),strides=(2,2),padding="same",
                                     use_bias=False,activation="tanh"))

    return model

# Discriminator
def build_discriminator():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64,(5,5),strides=(2,2),padding="same",
                            input_shape=[28,28,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128,(5,5),strides=(2,2),padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

generator = build_generator()
discriminator = build_discriminator()

# Loss + Optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)

# Train Step
@tf.function
def train_step(images):

    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_gen = gen_tape.gradient(gen_loss,
                                      generator.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss,
                                        discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gradients_gen,
                                      generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_disc,
                                       discriminator.trainable_variables))

    return gen_loss, disc_loss

# Save Generated Images
def save_images(epoch):
    noise = tf.random.normal([16, NOISE_DIM])
    preds = generator(noise, training=False)
    preds = (preds*127.5+127.5).numpy()

    for i in range(preds.shape[0]):
        img = preds[i]   # keep shape (28,28,1)
        tf.keras.utils.save_img(
            f"{SAVE_DIR}/epoch_{epoch}_{i}.png",
            img,
            scale=False
        )


# Training Loop
def train(dataset, epochs):
    for epoch in range(epochs):
        for batch in dataset:
            g_loss, d_loss = train_step(batch)

        print(f"Epoch {epoch+1} | Gen Loss: {g_loss:.4f} | Disc Loss: {d_loss:.4f}")
        mlflow.log_metric("generator_loss", float(g_loss), step=epoch)
        mlflow.log_metric("discriminator_loss", float(d_loss), step=epoch)
        save_images(epoch+1)

with mlflow.start_run():

    mlflow.set_tag("student_id", "202201761")

    mlflow.log_params({
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "noise_dim": NOISE_DIM
    })

    train(dataset, EPOCHS)

    mlflow.tensorflow.log_model(generator, "generator_model")
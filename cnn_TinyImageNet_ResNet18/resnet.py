import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow_datasets as tfds

# ==========================================================
# 1. Load Tiny ImageNet (from tfds)
# ==========================================================
dataset_name = 'tiny_imagenet200'
(ds_train, ds_val), ds_info = tfds.load(
    dataset_name,
    split=['train', 'validation'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

IMG_SIZE = 64
BATCH_SIZE = 128
NUM_CLASSES = ds_info.features['label'].num_classes

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = ds_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==========================================================
# 2. Define ResNet18 manually (light version)
# ==========================================================
def conv_block(x, filters, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Adjust shortcut if shape mismatch
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet18(input_shape=(64, 64, 3), num_classes=200):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Residual blocks
    x = conv_block(x, 64)
    x = conv_block(x, 64)

    x = conv_block(x, 128, stride=2)
    x = conv_block(x, 128)

    x = conv_block(x, 256, stride=2)
    x = conv_block(x, 256)

    x = conv_block(x, 512, stride=2)
    x = conv_block(x, 512)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model

model = build_resnet18()
model.summary()

# ==========================================================
# 3. Compile & Train
# ==========================================================
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10  # 可以从 10 开始快速实验
)

# ==========================================================
# 4. Save the model (new .keras format)
# ==========================================================
model.save("resnet18_tiny_imagenet.keras")

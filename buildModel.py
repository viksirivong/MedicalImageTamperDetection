import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import sys

MODEL_NAME            = 'tamperDetection.h5'

TRAIN_LOAD_PATH       = None
DEVTEST_LOAD_PATH     = None
DEFAULT_SAVE_PATH     = None

INPUT_SHAPE           = (64, 128, 128)      # 3D CT scans of dimension 64 slices x 128 length x 128 width
INITIAL_LEARNING_RATE = 0.0001              # initial learning rate.
NUM_EPOCHS            = 20                  # number of epochs.
BATCH_SIZE            = 2                   # batch size.

# preprocess step consists only of expanding across one dimension.
# data augmentation was not performed as it isn't useful for our application.
def preprocess(data, label):
    data = tf.expand_dims(data, axis=3)
    return data, label

def main():
    # if load path is not provided, the default one shall be used.
    load_path = sys.argv[1] if len(sys.argv) >= 2 else TRAIN_LOAD_PATH
    # if save path is not provided, the default one shall be used.
    save_path = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_SAVE_PATH

    # load training data.
    with np.load(load_path + '\\data.npz') as data:
        train_X = data['data_X']
        train_Y = data['data_Y']
 
    # load dev/test set
    with np.load(DEVTEST_LOAD_PATH + '\\devtest.npz') as data:
        devtest_X = data['devtest_X']
        devtest_Y = data['devtest_Y']
    
    # partition test set.
    test_X = devtest_X[:40] 
    test_Y = devtest_Y[:40] 

    # partition validation set.
    val_X = devtest_X[40:]
    val_Y = devtest_Y[40:]

    # create tensor datasets.
    train_dataset = tf.data.Dataset.from_tensor_slices( (train_X, train_Y) )
    val_dataset   = tf.data.Dataset.from_tensor_slices( (val_X, val_Y) )

    # preprocess training set.
    train_dataset = (
        train_dataset.shuffle(len(train_dataset))
        .map(preprocess)
        .batch(BATCH_SIZE)
    )
    # preprocess validation set.
    val_dataset = (
        val_dataset.shuffle(len(val_dataset))
        .map(preprocess)
        .batch(BATCH_SIZE)
    )

    # load model if one already exists.
    model = None
    if MODEL_NAME:
        model = tf.keras.models.load_model(MODEL_NAME)
    else:
        # Convolutional Neural Network Model
        model = tf.keras.Sequential([
                    tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu"),
                    tf.keras.layers.MaxPool3D(pool_size=2),
                    tf.keras.layers.BatchNormalization(),
                    
                    tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu"),
                    tf.keras.layers.MaxPool3D(pool_size=2),
                    tf.keras.layers.BatchNormalization(),

                    tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu"),
                    tf.keras.layers.MaxPool3D(pool_size=2),
                    tf.keras.layers.BatchNormalization(),

                    tf.keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu"),
                    tf.keras.layers.MaxPool3D(pool_size=2),
                    tf.keras.layers.BatchNormalization(),

                    tf.keras.layers.GlobalAveragePooling3D(),
                    tf.keras.layers.Dense(512, activation='relu'),
                    tf.keras.layers.Dropout(0.3),

                    tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
    # define learning rate schedule.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        INITIAL_LEARNING_RATE, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    # compile model.
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=['acc'],
    )
    
    # define callbacks.
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        "tamperDetection.h5", save_best_only=True
    )

    # define early stopping callback.
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

    # fit model to data.
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=NUM_EPOCHS,
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )

    # evaluate model on test set.
    model.evaluate(test_X, test_Y)

    # print model summary.
    print(model.summary())

    # plot model accuracy and loss for training/validation sets.
    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()
 
    for i, metric in enumerate(['acc', 'loss']):
        ax[i].plot(model.history.history[metric])
        ax[i].plot(model.history.history['val_' + metric])
        ax[i].set_title("Model {}".format(metric))
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metric)
        ax[i].legend(["training", "validation"])

    # save data as .npz in save_path
    # when loading this .npz file, use array['test_dataset'] for access
    np.savez_compressed(save_path + "\\testset.npz", test_X=np.array(test_X), test_Y=np.array(test_Y))

if __name__ == "__main__":
    main()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner import Hyperband

from config import Config


class CNNClassifier:
    def __init__(self, train_ds, val_ds, num_classes):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.num_classes = num_classes

    def build_model(self, hp):
        model = keras.Sequential(
#           layers.Rescaling(1./255, input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 3) )
        )

        model.add(layers.Conv2D(
            filters=32, kernel_size=(3, 3), activation='relu', padding='same',
            input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 3)
        ))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Flatten())

        model.add(layers.Dense(
            units=hp.Int('units', min_value=64, max_value=256, step=64),
            activation='relu'
        ))
        model.add(layers.BatchNormalization())
        
        model.add(layers.Dropout(
            rate=hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        ))

        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def hyperparameter_tuning(self):
        # ref: https://arxiv.org/pdf/1603.06560.pdf for Hyperband

        tuner = Hyperband(
            self.build_model,
            objective='val_accuracy',
            max_epochs=Config.MAX_EPOCHS_HP,
            factor=Config.FACTOR,
            hyperband_iterations=Config.HYPERBAND_ITERATIONS,
            directory=Config.hp_tuning_path,
            project_name=('re-prods' + Config.SUFFIX))

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        tuner.search(self.train_ds,
                     epochs=Config.EPOCHS,
                     validation_data=self.val_ds,
                     callbacks=[stop_early])

        best_model = tuner.get_best_models(num_models=1)[0]

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {best_hps.get('units')}. The optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')}. The optimal dropout rate is {best_hps.get('dropout_rate')}.
        """)

        return best_model

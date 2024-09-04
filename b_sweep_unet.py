import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cids
import wandb
from tensorflow import keras # type: ignore
from kadi_ai import KadiAIProject
from wandb.integration.keras.callbacks import WandbMetricsLogger


def doubleconv(x, filters, kernel_size=3, dropout=0.1):
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)
    return x


def downsample(x, filters, kernel_size=3, dropout=0.1):
    skip_connection = doubleconv(x, filters, kernel_size, dropout)
    next_layer = keras.layers.MaxPooling2D(2)(
        skip_connection
    )  # No: Avg pool (OR: strided conv)
    return skip_connection, next_layer


def upsample(
    x,
    skip_connection,
    filters,
    kernel_size=3,
    dropout=0.1,
    strides=2,
    padding="same",
):
    x = keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding
    )(x)
    x = keras.layers.concatenate([x, skip_connection])
    next_layer = doubleconv(x, filters=filters, dropout=dropout)
    return next_layer


# UNET
def model_function(data_definition):
    num_layers = 3
    dropout = 0
    input = keras.layers.Input(shape=data_definition.input_shape[1:])

    # encoder
    encoder_layers = []
    next_layer = input
    for i in range(num_layers):
        skip_connection, next_layer = downsample(
            next_layer, filters=64 * (2**i), dropout=dropout
        )
        encoder_layers.append(skip_connection)
        
    # bottleneck
    next_layer = doubleconv(next_layer, filters=64 * (2**(i+1)), dropout=dropout)

    # decoder
    for i in range(num_layers-1):
        next_layer = upsample(
            next_layer,
            skip_connection=encoder_layers[-(i + 1)],
            filters=64*(2**-(i-3)),
            dropout=dropout,
        )

    next_layer = upsample(
        next_layer,
        skip_connection=encoder_layers[-(i + 2)],
        filters=64*(2**-(i-3)),
        dropout=dropout,
        padding='valid'
    )

    for i in range(7,0,-1):
        next_layer = keras.layers.Conv2D(
            filters=2**i, kernel_size=3, padding="same", activation="relu"
        )(next_layer)
        next_layer = keras.layers.BatchNormalization()(next_layer)
        next_layer = keras.layers.Dropout(dropout)(next_layer)

    output = keras.layers.Conv2D(
        filters=data_definition.output_shape[-1], 
        kernel_size=3, padding="same", activation=None
    )(next_layer)

    model = keras.Model(inputs=input, outputs=output)

    return model


def schedule_function(hp):
    schedule = {
        "count": [1, wandb.config.epochs],
        "learning_rate": wandb.config.learning_rate,
        "batch_size": wandb.config.batch_size,
    }
    return schedule


def train():
    run = wandb.init()
    model_name = run.name
    project_name = "crack_prop"
    project_dir = "path_to/project_dir"
    project = KadiAIProject(project_name, root=project_dir)
    train_samples, valid_samples, test_samples = project.get_split_datasets(
        shuffle=True, valid_split=0.15, test_split=0.05,
    )
    data_definition = project.data_definition
    data_definition.input_features = ['features_in_solid_angles', 'feature_crack_in']
    data_definition.output_features = ['feature_energy_out', 'crack_out']

    model = cids.CIDSModelTF(
        data_definition,
        model_function,
        name=model_name,
        identifier="",
        result_dir=project.result_dir,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        num_gpus=2,
    )

    model.online_normalize = True
    model.metrics.append("mae")
    model.monitor = "val_mae"
    model.loss = keras.losses.MeanSquaredError()
    model.encode_categorical = False
    model.VERBOSITY = 2
    model.save_best_only = True

    history = model.train(
                train_samples,
                valid_samples,
                # hp=hp,
                schedule=schedule_function,
                callbacks=[WandbMetricsLogger(), 
                           keras.callbacks.EarlyStopping(patience=5)],
    )

    wandb.log({
        "val_mae" : min(history["val_mae"])
    })
    

if __name__ == "__main__":
    sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_mae"},
    "parameters": {
        "batch_size": {"values": [8, 16, 32, 64, 128, 256]},
        "epochs": {"values": [25, 50, 75, 100]},
        "dropout": {"values": [0, 0.1, 0.2, 0.3]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        },
    }

    wandb.agent("sweep_id" , function=train, project="crack-prop", count=50)
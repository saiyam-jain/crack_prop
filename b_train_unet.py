import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import cids
import wandb
import argparse
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
        "count": [1, epochs],
        "learning_rate": 0.0001,
        "batch_size": batch_size,
    }
    return schedule


def main():

    wandb.init(
        name=model_name,
        # set the wandb project where this run will be logged
        project="crack-prop",
        config={
            "batch_size": batch_size,  # divided by number of GPUs
            "epochs": epochs,
            "model_name": model_name,
        },
    )
    wandb.run.log_code(".") # type: ignore

    # Data paths
    project_name = "crack_prop"
    project_dir = "path_to/project_dir"
    project = KadiAIProject(project_name, root=project_dir)

    # Read paths
    train_samples, valid_samples, test_samples = project.get_split_datasets(
        shuffle=True, valid_split=0.15, test_split=0.05,
    )

    # Train only with solid angles and learn energy in output
    data_definition = project.data_definition
    data_definition.input_features = ['features_in_solid_angles', 'feature_crack_in']
    data_definition.output_features = ['feature_energy_out', 'crack_out']

    model = cids.CIDSModelTF(
        data_definition,
        model_function,
        name=model_name,
        identifier="",
        result_dir=project.result_dir,
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        num_gpus=1,
    )

    model.online_normalize = True
    model.metrics.append("mae")
    model.monitor = "val_mae"
    model.loss = keras.losses.MeanSquaredError()
    model.encode_categorical = False
    model.VERBOSITY = 2
    model.save_best_only = True

    model.train(
        train_samples,
        valid_samples,
        # hp=hp,
        schedule=schedule_function,
        callbacks=[WandbMetricsLogger(),
                keras.callbacks.EarlyStopping(patience=50)],
    )

    project.log(">> Training complete.")
    model.save(os.path.join(project_dir, "model-" + model_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train crackprop UNet model")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size, default=16")
    parser.add_argument("--epochs", type=int, default=1000, help="epochs, default=500")
    parser.add_argument("--model_name", help="model_name")
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    model_name = args.model_name
    
    main()
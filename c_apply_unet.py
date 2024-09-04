import os
import glob
import cids
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras # type: ignore
from kadi_ai import KadiAIProject
from matplotlib.lines import Line2D
from pypace.core.data.scalardata import ScalarDataReader
from skimage.metrics import structural_similarity as ssim
from matplotlib.colors import ListedColormap

def read_infile_keys(infile_saved):
    # extract the domain size and anisotropy orientation from .infile_saved
    with open(infile_saved, "r") as infile:
        for line in infile.read().split("\n"):
            if "Settings.Domain.NumX" in line:
                numx = float(line.split("=")[1])

            if "Settings.Domain.NumY" in line:
                numy = float(line.split("=")[1])

            if "Phasefield.Crack.Brittle.Aniso.angles" in line:
                list_angles = []
                for angles in line.split("=")[1][2:-2].split("),("):
                    list_angles.append(float(angles.split(",")[2]))

        list_angles = np.asarray(list_angles, dtype=np.float32)
        list_angles += 50.0
        list_angles /= 100.0

    return numx, numy, list_angles


def get_crack_tip_xy_position(crack):
    # get crack tip x-y coordinates
    one_coordinates = np.where(crack==1)
    x = one_coordinates[1].max()
    y = one_coordinates[0][np.argmax(one_coordinates[1])]
    if np.argwhere(one_coordinates[1] == x).flatten().shape!=(1,):
        for j in np.argwhere(one_coordinates[1] == x):
            if y<one_coordinates[0][j[0]]:
                y=one_coordinates[0][j[0]]
    return x,y


def get_clean_cluster(image, start_coord):
    """
    Cluster the image starting from the given coordinate until encountering a zero.
    Any value outside the cluster is set to zero.
    
    Parameters:
    image (np.array): 2D array of floats with values in the range [0,1]
    start_coord (tuple): Starting coordinate (row, col) for clustering
    
    Returns:
    np.array: Clustered image
    """
    rows, cols = image.shape
    clustered_image = np.zeros((rows, cols))
    
    # Check if the starting coordinate is valid and non-zero
    if image[start_coord[0], start_coord[1]] == 0:
        return clustered_image
    
    # Stack for DFS
    stack = [start_coord]
    while stack:
        x, y = stack.pop()
        if clustered_image[x, y] == 0 and image[x, y] != 0:
            clustered_image[x, y] = image[x, y]
            # Add neighbors to stack if they are within bounds
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and clustered_image[nx, ny] == 0:
                    stack.append((nx, ny))
    
    return clustered_image

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


def calculate_mse(array1, array2):
    return np.mean((array1 - array2) ** 2)

def calculate_mae(array1, array2):
    return np.mean(np.abs(array1 - array2))

def calculate_ssim(array1, array2):
    return ssim(array1, array2, data_range=array2.max() - array2.min())

def main():
    project_name = "crack_prop"
    project_dir = "path_to/project_dir"
    project = KadiAIProject(project_name, root=project_dir)

    data_definition = project.data_definition
    data_definition.input_features = ['features_in_solid_angles', 'feature_crack_in']
    data_definition.output_features = ['feature_energy_out', 'crack_out']

    model = cids.CIDSModelTF(
        data_definition,
        model_function,
        name="model_name",
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

    path = "path_to/test_simulation"
    infile_saved = glob.glob(os.path.join(path, "*infile_saved"))[0]
    numx, numy, list_angles = read_infile_keys(infile_saved)

    crack_domain = ScalarDataReader(f"{path}/301x301_8_phases_crack_evolution_1.CRACK_crack.p3s").read(0)[1]

    phases_domain=[]
    for i in range(8):
        phase = ScalarDataReader(f"{path}/301x301_8_phases_crack_evolution_1.phi_phase{i+1}.p3s").read(0)[1]
        phases_domain.append(phase)

    crack_tip_x, crack_tip_y = get_crack_tip_xy_position(crack_domain)

    while crack_tip_x<280 and crack_tip_y<280 and crack_tip_y>21 and crack_tip_x>11:
        crack_tip_x, crack_tip_y = get_crack_tip_xy_position(crack_domain)
        print(crack_tip_x, crack_tip_y)
        y_min = crack_tip_y-20
        y_max = crack_tip_y+20+1
        x_min = crack_tip_x-10
        x_max = crack_tip_x+30+1

        crack_window = crack_domain[y_min:y_max, x_min:x_max].copy()
        
        solid_field_stack = []
        features_in_solid_angles = np.zeros((41,41))

        for phase in phases_domain:
            solid_field_stack.append(phase[y_min:y_max, x_min:x_max])

        solid_field_stack = np.asarray(solid_field_stack)
        solid_field_stack = solid_field_stack.sum(axis=0)

        for i, phase in enumerate(phases_domain):
            phase_window = phase[y_min:y_max, x_min:x_max]
            features_in_solid_angles += list_angles[i] * phase_window / solid_field_stack

        features_in_solid_angles = np.asarray(np.nan_to_num(features_in_solid_angles, nan=-1.0))

        features = np.zeros((1, 41, 41, 2))

        features[0, :, :, 0] = features_in_solid_angles
        features[0, :, :, 1] = crack_window

        phi_c = model.predict(features, use_gpu=True)

        crack_window_predicted = phi_c[0,...,-1] # type: ignore
        crack_window_predicted = (crack_window_predicted - crack_window_predicted.min())/(crack_window_predicted.max()-crack_window_predicted.min())
        crack_window_predicted = np.where(crack_window_predicted > 0.96, 1.0, crack_window_predicted)
        crack_window_predicted = np.where(crack_window_predicted < 0.04, 0.0, crack_window_predicted) # type: ignore
        crack_window_predicted = get_clean_cluster(crack_window_predicted, (20,10))

        crack_domain[y_min:y_max, x_min+10:x_max] = crack_window_predicted[:, 10:].copy()

        for i in range(8):
            phases_domain[i] = phases_domain[i] - crack_domain * phases_domain[i]

    np.save("path", crack_domain)
    
    phiindex = ScalarDataReader("path.phiindex.p3s").read(0)[1]
    crack_gt = ScalarDataReader("path.CRACK_crack.p3s").read(0)[1]
    print(calculate_mse(crack_gt, crack_domain), calculate_mae(crack_gt, crack_domain), calculate_ssim(crack_gt, crack_domain))
    
    mask = crack_domain > 0.95
    overlay = np.where(mask, -1, phiindex)
    plasma = plt.cm.plasma(np.linspace(0, 1, 8)) # type: ignore
    custom_colors = np.vstack(([[0, 0, 0, 1]], plasma, [[1, 0, 0, 1]]))  # Black for -1 and Red for 8
    cmap = ListedColormap(custom_colors)

    custom_lines = [Line2D([0], [0], color=(1, 0, 0, 1), lw=4),
                    Line2D([0], [0], color=(0, 0, 0, 1), lw=4)]

    plt.figure(figsize=(10, 8))
    plt.imshow(overlay, cmap=cmap)
    plt.legend(custom_lines, ['GT', 'Pred'])
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.savefig("path_output_crack_evolution_1", dpi=1)


if __name__ == "__main__":
    main()
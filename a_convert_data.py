import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import glob
import tqdm
import subprocess
import numpy as np
import tensorflow as tf
from pathlib import Path
from cids.data import Feature
from cids.data import DataWriter
from kadi_ai import KadiAIProject
from pypace.core import PaceSimulation
from cids.data import DataDefinition
from pypace.core.data.scalardata import ScalarDataReader

# Angles are not normalized between 0 and 1
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
        # list_angles += 50.0
        # list_angles /= 100.0

    return numx, numy, list_angles


data_type = np.float32
Nx = 41
Ny = 41

# Data paths
project_name = "crack_prop"
project_dir = "path_to/project_dir"
project = KadiAIProject(project_name, root=project_dir)

# Project creates an `input_dir` in the `project_dir`, which stores converted
# input data as tfrecords in a subdirectory `tfrecord`
tfrecord_dir = Path(project.input_dir) / "tfrecord"

# Data definition
data_definition = DataDefinition(
    Feature(
        "features_in_solid_angles",
        [None, 41, 41, 1],
        data_format = "NXYF",
        dtype = tf.string, # type: ignore
        decode_str_to = tf.float32, # type: ignore
    ),
    Feature(
        "feature_crack_in",
        [None, 41, 41, 1],
        data_format = "NXYF",
        dtype = tf.string, # type: ignore
        decode_str_to = tf.float32, # type: ignore
    ),
    Feature(
        "feature_j_integral_in",
        [None, 2],
        data_format = "NF",
        dtype = tf.string, # type: ignore
        decode_str_to = tf.float32, # type: ignore
    ),
    Feature(
        "feature_Ux_in",
        [None, 41, 41, 1],
        data_format = "NXYF",
        dtype = tf.string, # type: ignore
        decode_str_to = tf.float32, # type: ignore
    ),
    Feature(
        "feature_Uy_in",
        [None, 41, 41, 1],
        data_format = "NXYF",
        dtype = tf.string, # type: ignore
        decode_str_to = tf.float32, # type: ignore
    ),
    Feature(
        "feature_energy_in",
        [None, 41, 41, 1],
        data_format = "NXYF",
        dtype = tf.string, # type: ignore
        decode_str_to = tf.float32, # type: ignore
    ),
    Feature(
        "feature_in_elasticity_stress",
        [None, 41, 41, 3],
        data_format = "NXYF",
        dtype = tf.string, # type: ignore
        decode_str_to = tf.float32, # type: ignore
    ),
    Feature(
        "feature_Uy_out",
        [None, 41, 41, 1],
        data_format = "NXYF",
        dtype = tf.string, # type: ignore
        decode_str_to = tf.float32, # type: ignore
    ),
    Feature(
        "feature_Ux_out",
        [None, 41, 41, 1],
        data_format = "NXYF",
        dtype = tf.string, # type: ignore
        decode_str_to = tf.float32, # type: ignore
    ),
    Feature(
        "feature_energy_out",
        [None, 41, 41, 1],
        data_format = "NXYF",
        dtype = tf.string, # type: ignore
        decode_str_to = tf.float32, # type: ignore
    ),
    Feature(
        "feature_j_integral_out",
        [None, 2],
        data_format = "NF",
        dtype = tf.string, # type: ignore
        decode_str_to = tf.float32, # type: ignore
    ),
    Feature(
        "feature_out_elasticity_stress",
        [None, 41, 41, 3],
        data_format = "NXYF",
        dtype = tf.string, # type: ignore
        decode_str_to = tf.float32, # type: ignore
    ),
    Feature(
        "crack_out", 
        [None, 41, 41, 1], 
        data_format = "NXYF", 
        dtype = tf.string, # type: ignore
        decode_str_to = tf.float32 # type: ignore
    ),
    dtype=tf.float32, # type: ignore
)

project.data_definition = data_definition

# Create a data converter object
data_writer = DataWriter(data_definition)

list_simulations = [simulation for simulation in
                    glob.glob(os.path.join("path_to/evolution_dir",
                                           "*evolution*")) if os.path.isdir(simulation)]

# for each simulation
for simulation in tqdm.tqdm(list_simulations):
    print(simulation)

    # if no information of the anisotropy orientation is available, skip the simulation
    try:
        infile_saved = glob.glob(os.path.join(simulation, "*infile_saved"))[0]
    except Exception:
        print("infile_saved not found")
        continue

    numx, numy, list_angles = read_infile_keys(infile_saved)

    # extract subwindows from zip file
    if not os.path.isdir(os.path.join(simulation, "subwindow_20")):
        subprocess.run(["tar", "-zxf", "subwindow.tar.gz"], cwd=f"{simulation}")

    list_path_base_files_windows = [file.split(".")[0] for file in 
                                    glob.glob(os.path.join(simulation, "subwindow_20", "*frame*CRACK_crack.p3s"))]
    
    for path_base_file_window in list_path_base_files_windows:
        features = {}

        base_file_window = path_base_file_window.split("/")[-1]

        # read the data for subwindow
        try:
            pace_window = PaceSimulation(path_base_file_window + ".p3simgeo")
            pace_window.add_all_fields()
        except Exception as error:
            print(error)
            continue

        # if second frame does not exist, the input data can not be mapped to an output
        if ScalarDataReader(f"{path_base_file_window}.CRACK_crack.p3s").frame_count == 1:
            print(f"{path_base_file_window}.CRACK_crack.p3s is missing or has only one frame")
            continue

        # checking if all solid phases exist
        all_files_exist = True
        for index in range(8):
            file = f"{path_base_file_window}.phi_phase{index + 1}.p3s"
            if not os.path.exists(file) or ScalarDataReader(file).frame_count!=2:
                print(f"File {file} missing or has less than 2 frames.")
                all_files_exist = False

        # checking if all stress fields exists
        for i in 11, 12, 22:
            file = f"{path_base_file_window}.elasticity_stress{i}.p3s"
            if not os.path.exists(file) or ScalarDataReader(file).frame_count!=2:
                print(f"File {file} missing or has less than 2 frames.")
                all_files_exist = False

        # checking if elasticity_energy exists
        file = f"{path_base_file_window}.elasticity_energy.p3s"
        if not os.path.exists(file) or ScalarDataReader(file).frame_count!=2:
            print(f"File {file} missing or has less than 2 frames.")
            all_files_exist = False

        # checking if elasticity_Ux exists
        file = f"{path_base_file_window}.elasticity_Ux.p3s"
        if not os.path.exists(file) or ScalarDataReader(file).frame_count!=2:
            print(f"File {file} missing or has less than 2 frames.")
            all_files_exist = False

        # checking if elasticity_Uy exists
        file = f"{path_base_file_window}.elasticity_Uy.p3s"
        if not os.path.exists(file) or ScalarDataReader(file).frame_count!=2:
            print(f"File {file} missing or has less than 2 frames.")
            all_files_exist = False

        # checking if j_integral exists
        file = f"{path_base_file_window}_j_integral.npy"
        if not os.path.exists(file):
            print(f"File {file} missing.")
            all_files_exist = False

        if not all_files_exist:
            continue

        solid_field_stack = []

        for index, field in enumerate(sorted(pace_window.fields)):
            file = f"{path_base_file_window}.{field}.p3s"

            if field in [f'phi_phase{i+1}' for i in range(8)]:
                field_reader = ScalarDataReader(file)
                solid_field_stack.append(field_reader.read(0)[0])

        solid_field_stack = np.asarray(solid_field_stack, dtype=data_type)
        solid_field_stack = solid_field_stack.sum(axis=0)

        features_in_solid_angles = np.zeros((Ny, Nx))

        angle_count = 0
        for index, field in enumerate(sorted(pace_window.fields)):
            file = f"{path_base_file_window}.{field}.p3s"

            if field in [f'phi_phase{i+1}' for i in range(8)]:
                field_reader = ScalarDataReader(file)
                features_in_solid_angles += list_angles[angle_count] * field_reader.read(0)[0] / solid_field_stack
                angle_count+=1
                
        # nan assigned to 0
        features_in_solid_angles = np.nan_to_num(features_in_solid_angles, nan=0)
        features_in_solid_angles = np.asarray(features_in_solid_angles, dtype=data_type)

        feature_crack = ScalarDataReader(f"{path_base_file_window}.CRACK_crack.p3s")
        
        # CRACK_crack, frame = 0 as input
        feature_crack_in = np.asarray(feature_crack.read(0)[0], dtype=data_type)

        feature_crack_in = np.asarray(feature_crack_in, dtype=data_type)

        if np.isnan(features_in_solid_angles.sum() or feature_crack_in.sum()):
            print(f"Is nan: {path_base_file_window}")
            continue

        stress11 = ScalarDataReader(f"{path_base_file_window}.elasticity_stress11.p3s")
        stress12 = ScalarDataReader(f"{path_base_file_window}.elasticity_stress12.p3s")
        stress22 = ScalarDataReader(f"{path_base_file_window}.elasticity_stress22.p3s")
        energy = ScalarDataReader(f"{path_base_file_window}.elasticity_energy.p3s")
        Ux = ScalarDataReader(f"{path_base_file_window}.elasticity_Ux.p3s")
        Uy = ScalarDataReader(f"{path_base_file_window}.elasticity_Uy.p3s")
        try:
            j_integral = np.load(f"{path_base_file_window}_j_integral.npy", allow_pickle=True)
        except Exception as error:
            print(error)
            continue

        # frame = 0 as input
        feature_in_elasticity_stress = np.stack([stress11.read(0)[0], 
                                                    stress12.read(0)[0], 
                                                    stress22.read(0)[0]], 
                                                    axis=-1,
                                                    dtype=data_type)
        
        # frame = 1 as output
        feature_out_elasticity_stress = np.stack([stress11.read(1)[0],
                                                    stress12.read(1)[0], 
                                                    stress22.read(1)[0]], 
                                                    axis=-1,
                                                    dtype=data_type)
        
        
        features['features_in_solid_angles'] = features_in_solid_angles
        features['feature_crack_in'] = feature_crack_in
        features['feature_in_elasticity_stress'] = feature_in_elasticity_stress
        features['feature_energy_in'] = energy.read(0)[0]
        features['feature_Ux_in'] = Ux.read(0)[0]
        features['feature_Uy_in'] = Uy.read(0)[0]
        features['feature_j_integral_in'] = j_integral[0, 0:2]

        # CRACK_crack, frame = 1 as output
        feature_out = np.asarray(feature_crack.read(1)[0], dtype=data_type)
        
        if np.isnan(feature_out.sum()):
            print(f"Is nan: {path_base_file_window}")
            continue

        features["crack_out"] = feature_out  # for z = 0 and t = 1
        features['feature_out_elasticity_stress'] = feature_out_elasticity_stress
        features['feature_energy_out'] = energy.read(1)[0]
        features['feature_Ux_out'] = Ux.read(1)[0]
        features['feature_Uy_out'] = Uy.read(1)[0]
        features['feature_j_integral_out'] = j_integral[1, 0:2]

        out_file = tfrecord_dir / f"{base_file_window}.tfrecord"

        # Write features to file
        try:
            data_writer.write_example(out_file, features)
        except KeyError as e:
            project.warn(f"Missing key {e.args[0]} in: {os.fspath(out_file)}")
            continue

    subprocess.run(["rm",
            "-r", 
            "subwindow_20"],
            cwd=f"{simulation}")


project.log(f"Done processing")

# Write the data definition and the features to a human-readable json file
#   The json file can also be loaded directly later-on for training.
project.data_definition = data_definition
project.to_json(write_data_definition=True)

project.log("Done.")
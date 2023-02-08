import os

import torch
# Define weather or not to use cuda (Graphic card only!)
# Sets the device where the processing should be done

device = "cuda" if torch.cuda.is_available() else "cpu"

preload_to_gpu = True

preload_device = device if preload_to_gpu else "cpu"

print("Using {} device. Data gets loaded to {}.".format(device, preload_device))

root_path = __file__[:-17]

skip_too_many_perms = True

auto_encoder_training_intermediate_layer_size = 400
auto_encoder_encoding_size = 100


def get_cuda_memory_info():
    if (device == 'cpu'):
        return "-"
    t = round(torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 2))
    r = round(torch.cuda.memory_reserved(0) / (1024.0 ** 2))
    a = round(torch.cuda.memory_allocated(0) / (1024.0 ** 2))
    return f"{a}/{r}/{t} MiB"
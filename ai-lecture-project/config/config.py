import torch
# Define weather or not to use cuda (Graphic card only!)
# Sets the device where the processing should be done
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device.".format(device))

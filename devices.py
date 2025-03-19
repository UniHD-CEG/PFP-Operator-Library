import tvm

valid_devices = ["pi3", "pi4","pi5"]

def get_device(input:str):

    index = valid_devices.index(input)
    if not (input in valid_devices):
        raise ValueError(input + " is not a valid device name! Supported devices names:" + str(valid_devices))

    if valid_devices[index] == "pi3":
        target=str(tvm.target.arm_cpu('rasp3b'))  + " -mfloat-abi=hard" 
        device_key='pi-cluster-node-pi3b-unity'
        device_type = "cpu"
    elif valid_devices[index] == "pi4":
        target=tvm.target.arm_cpu('rasp4b64')
        device_key='pi-cluster-node-pi4b-unity'
        device_type = "cpu"
    elif valid_devices[index] == "pi5":
        target="llvm -device=arm_cpu -model=bcm2712 -mtriple=aarch64-linux-gnu -mattr=+neon -mcpu=cortex-a76" 
        device_key='pi-cluster-node-pi5-unity'
        device_type = "cpu"
    else: # should never happen!
        raise ValueError("No information for device: " + input)

    return target, device_key, device_type

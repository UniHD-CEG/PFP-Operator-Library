from tvm import rpc
from tvm.contrib import utils


"""
Usage (either): 
1) module_upload(device_key) directly then run
2)
session = get_session(device_key)
get_device(session)
module_upload(session)
if additional logic with device is needed before uploading the module 
"""

class rpc_config:
    host = "0.0.0.0"
    port = 9000
    timeout = 25000

def get_session(device_key: str) -> rpc.TrackerSession:
    """
    connects to the device identified by "device_key", returns RPC-session
    """
    tracker = rpc.connect_tracker(rpc_config.host, rpc_config.port)
    remote = tracker.request(device_key, session_timeout=rpc_config.timeout)
    return remote


def get_device(session: rpc.TrackerSession = None, device_type: str = "cpu"):
    """
    returns the desired device (cpu or gpu) for a given session
    """
    dev = session.cpu(0) if device_type == "cpu" else session.cuda(0)
    return dev


def module_upload(mod, device_key: str = None, session: rpc.TrackerSession = None, device_type: str = "cpu"):
    """
    uploads a library to a device indetified by its key or an already running session
    """
    module_upload.counter +=1 # only used if a single program uploads more than one lib per call
    if (device_key is None and session is None) or (device_key is not None and session is not None):
        raise ValueError('Can only connect with key OR session')
    remote = get_session(device_key) if session is None else session
    remote_device = get_device(session=remote, device_type=device_type)

    # Upload module
    tmp = utils.tempdir()
    path = tmp.relpath("mod"+ str(module_upload.counter)+".tar")

    if hasattr(mod, 'export_library'): # this works for Relax or TensorIR modules
        mod.export_library(path)
    elif hasattr(mod.mod, 'export_library'): # Relay VM executable has mod attribute that is the actual module
        mod.mod.export_library(path)
    else:
        raise TypeError("Type not supported by module_upload" + str(type(mod)))
    remote.upload(path)
    remote_module = remote.load_module("mod"+ str(module_upload.counter)+".tar")

    return remote_module, remote_device
module_upload.counter = 0

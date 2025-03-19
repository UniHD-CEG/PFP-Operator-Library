import numpy as np
import torch


# weight format convert pyro VI saved model (log var) to PFP (var rho)
def convert_logvar_to_var_rho(var, transpose=False):
    if transpose:
        vars_norm = torch.exp(var.detach().T)
    else:
        vars_norm = torch.exp(var.detach())
    var_rho = np.log( np.expm1( vars_norm ) )
    return var_rho

# pyro VI saved to plain variance
def convert_logvar_to_var(var, transpose=False):
    if transpose:
        vars_norm = torch.exp(var.detach().T)
    else:
        vars_norm = torch.exp(var.detach())
    return vars_norm

def rescale_variances(variance, variance_rescale_factor):
    variance *= variance_rescale_factor
    return variance

# w_var --> E[w^2]
def convert_var_to_second_raw_moment(mean, var):
    return mean * mean + var

def load_from_pyro_dict(model, layer_name, var_type, weight_type, guide_name='ScaledAutoNormal', transpose=False, convert_logvars_to_var_rho=False, convert_logvars_to_var=True, to_numpy=False, variance_rescale_factor=1.0):
    """
    Pick values from pyro dict (pretrained VI model)

    :param model: pyro_state_dict
    :param layer_name: examples input_layer, out_layer, fc1, hidden_layers.{4}, ...
    :param var_type: mean or variance in pyro notation 'locs' or 'scales'
    :param weight_type: weight or bias
    :param guide_name: typically AutoNormal or ScaledAutoNormal
    """
    key_str = f"{guide_name}.{var_type}.{layer_name}.{weight_type}"
    value = model['params'][key_str]

    # conversion between formats
    if convert_logvars_to_var_rho and var_type=='scales':
        value = convert_logvar_to_var_rho(value, transpose=False)
    elif convert_logvars_to_var and var_type=='scales':
        value = convert_logvar_to_var(value, transpose=False)

    # rescale variances with a factor
    if var_type=='scales' and variance_rescale_factor != 1.0:
        value = rescale_variances(value, variance_rescale_factor)

    if transpose:
        value = value.T

    if to_numpy:
        value = value.detach().cpu().numpy()

    return value 


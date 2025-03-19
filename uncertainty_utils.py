import numpy as np
import torch
from torcheval.metrics import BinaryAUROC

def softmax_entropy(logits: torch.Tensor, dim=2):
    """
    Compute softmax entropy of classification for aleatoric uncertainty like in
    https://arxiv.org/pdf/2102.11582.pdf
    https://people.csail.mit.edu/lrchai/files/Chai_thesis.pdf
    :param logits: NN output
    :return: softmax entropy per sample
    """
    probs = torch.nn.functional.softmax(logits, dim=dim)
    log_probs = torch.nn.functional.log_softmax(logits, dim=dim)

    entropy = - torch.mean(torch.sum(probs * log_probs, dim=dim), dim=0)
    return entropy

def softmax_entropy_no_sample_dim(logits: torch.Tensor):
    """
    Compute softmax entropy of classification for aleatoric uncertainty like in
    https://arxiv.org/pdf/2102.11582.pdf
    https://people.csail.mit.edu/lrchai/files/Chai_thesis.pdf
    This is a version of the classical softmax entropy but since with the PFP 
    sampling is not necessary the outer dimension to average over is missing.
    :param logits: NN output
    :return: softmax entropy per sample
    """
    probs = torch.nn.functional.softmax(logits, dim=1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)

    entropy = - torch.sum(probs * log_probs, dim=1)
    return entropy

def predictive_uncertainty(logits: torch.Tensor, dim=2):
    """
    https://people.csail.mit.edu/lrchai/files/Chai_thesis.pdf
    https://www.cs.ox.ac.uk/people/yarin.gal/website/thesis/thesis.pdf
    :param logits: NN output
    :return: predictive uncertainty per sample
    """
    eps = 1e-12
    probs = torch.nn.functional.softmax(logits, dim=dim)

    pred_uncertainty = - torch.sum(torch.mean(probs, dim=0) * torch.log(torch.mean(probs, dim=0) + eps), dim=dim-1)
    return pred_uncertainty

########### AUROC ################
# we use a binary AUROC here, to classify how good the epistemic uncertainty measure
# can distinguish between in-domaini (ID) and out-of-domain data (OOD)
def calculate_AUROC(epistemic_uncertainty, isOOD):
    """
        epistemic_uncertainty: some measure of epistemic uncertainty, normally Mutual Information of VI, or avg. Variance of the PFP
        isOOD: binary truth for OOD = 1, for ID = 0
    """
    metric = BinaryAUROC()
    metric.update(epistemic_uncertainty, isOOD)
    return metric.compute()

# for the dirtyMNIST case we have ID or OOD information separated for the datasets
def calculate_AUROC_for_datasets( list_of_tuples_with_epistemic_uncertainty_and_isOOD_flag ):
    """
    provide a list of tuples [(epistemic uncertainty for dataset A, bool is dataset A OOD),....]
    """
    list_ec = []
    list_isOOD = []
    for epistemic_uncertainty, isOODflag in list_of_tuples_with_epistemic_uncertainty_and_isOOD_flag:
        list_ec.append(epistemic_uncertainty)
        if isOODflag:
            isOODtensor = torch.ones_like(epistemic_uncertainty)
        else:
            isOODtensor = torch.zeros_like(epistemic_uncertainty)
        list_isOOD.append(isOODtensor)
    ec = torch.cat(list_ec)
    isOOD = torch.cat(list_isOOD)
    auroc = calculate_AUROC(ec, isOOD)
    return auroc

# this script specific function to calc and store the auroc
def calculate_AUROC_TVMscript(variances_dict):
    list_to_calculate_auroc = [
        (torch.from_numpy(variances_dict['MNIST'].mean(axis=1)),False),
        (torch.from_numpy(variances_dict['AmbiguousMNIST'].mean(axis=1)),False),
        (torch.from_numpy(variances_dict['FashionMNIST'].mean(axis=1)),True),
    ]
    auroc = calculate_AUROC_for_datasets( list_to_calculate_auroc )
    return auroc



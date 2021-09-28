import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_model

from utils.torch_utils import gen_rand_tensor
from utils.bo_attack import evaluate

def gen_initial_data(i, train_param, img):
    """
    arg:
        d: dimension of lower dimensional perturbation (i.e. delta \in d^r)
        target_img: image being targeted for attack
        target_label: original label of the target img
        n: number of initial observations
    """
    # range of parameter period_x,y and freq is [0, d] for d is side-length of img
    # get the min of image, img.shape = (C,H,W), side length

    train_param[i, 0] = gen_rand_tensor(low=20.0, high=160.0, size=(1, ), dtype=img.dtype).to(img) # period_x
    train_param[i, 1] = gen_rand_tensor(low=20.0, high=160.0, size=(1, ), dtype=img.dtype).to(img) # period_y
    train_param[i, 2] = gen_rand_tensor(low=0.51, high=4.5, size=(1, ), dtype=img.dtype).to(img) # octaves
    train_param[i, 3] = gen_rand_tensor(low=4.0, high=32.0, size=(1, ), dtype=img.dtype).to(img)  # freq

    return train_param


def optimize_acqf_and_get_observation(acq_func, img, detector, model, device, dtype, imgsz=640, norm=16, n=3, NUM_RESTARTS=10, RAW_SAMPLES=100):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation"""
    # range of parameter period_x,y and freq is [0, d] for d is side-length of img
    # get the min of image, img.shape = (C,H,W), side length
    d = min(img.shape[1:])
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.stack([
            torch.tensor([20.0, 20.0, 0.51, 4.0], device=device, dtype=dtype),
            torch.tensor([160.0, 160.0, 4.5, 32.0], device=device, dtype=dtype),
        ]),
        q=n,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )

    # observe new values 
    new_x = candidates.detach()
    pred, new_obj = evaluate(img, detector, model, new_x[:,0], new_x[:,1], new_x[:,2], new_x[:,3], imgsz=imgsz, norm=norm)
    return new_x, new_obj, pred


def get_fitted_model(train_x, train_obj, state_dict=None):
    # initialize and fit model
    model = SingleTaskGP(train_X=train_x, train_Y=train_obj)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.to(train_x)
    fit_gpytorch_model(mll)
    return model
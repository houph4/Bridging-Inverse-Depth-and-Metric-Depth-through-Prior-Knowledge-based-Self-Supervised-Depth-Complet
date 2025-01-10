import numpy as np


def root_mean_sq_err(src, tgt,mask=None):
    '''
    Root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : root mean squared error
    '''
    if mask is not None:
        # Ensure mask is boolean for indexing
        mask = mask.astype(bool)
        # Apply the mask
        src = src[mask]
        tgt = tgt[mask]

    return np.sqrt(np.mean((tgt - src) ** 2))


def mean_abs_err(src, tgt, mask=None):
    '''
    Mean absolute error with optional mask

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
        mask : numpy[bool] or numpy[int], optional
            binary mask where 1 (or True) indicates the region to consider for error computation
    Returns:
        float : mean absolute error in the masked region
    '''
    if mask is not None:
        # Ensure mask is boolean for indexing
        mask = mask.astype(bool)
        # Apply the mask
        src = src[mask]
        tgt = tgt[mask]

    return np.mean(np.abs(tgt - src))

def inv_root_mean_sq_err(src, tgt, mask=None):
    '''
    Inverse root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
        mask : numpy[bool] or numpy[int], optional
            binary mask where 1 (or True) indicates the region to consider for error computation
    Returns:
        float : inverse root mean squared error
    '''
    if mask is not None:
        # Ensure mask is boolean for indexing
        mask = mask.astype(bool)
        # Apply the mask
        src = src[mask]
        tgt = tgt[mask]

    return np.sqrt(np.mean(((1.0 / tgt) - (1.0 / src)) ** 2))


def inv_mean_abs_err(src, tgt, mask=None):
    '''
    Inverse mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
        mask : numpy[bool] or numpy[int], optional
            binary mask where 1 (or True) indicates the region to consider for error computation
    Returns:
        float : inverse mean absolute error
    '''
    if mask is not None:
        # Ensure mask is boolean for indexing
        mask = mask.astype(bool)
        # Apply the mask
        src = src[mask]
        tgt = tgt[mask]

    return np.mean(np.abs((1.0 / tgt) - (1.0 / src)))


def mean_abs_rel_err(src, tgt, mask=None):
    '''
    Mean absolute relative error (normalize absolute error)

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
        mask : numpy[bool] or numpy[int], optional
            binary mask where 1 (or True) indicates the region to consider for error computation
    Returns:
        float : mean absolute relative error between source and target
    '''
    if mask is not None:
        # Ensure mask is boolean for indexing
        mask = mask.astype(bool)
        # Apply the mask
        src = src[mask]
        tgt = tgt[mask]

    return np.mean(np.abs(src - tgt) / tgt)

def compute_errors(gt, pred, mask=None):
    """
    Computation of error metrics between predicted and ground truth depths with optional mask.

    Arg(s):
        gt : numpy[float32]
            Ground truth depth array
        pred : numpy[float32]
            Predicted depth array
        mask : numpy[bool] or numpy[int], optional
            Binary mask where 1 (or True) indicates the region to consider for error computation

    Returns:
        tuple : error metrics (abs_rel, sq_rel, rmse, rmse_log, inv_rmse, inv_mae, a1, a2, a3)
    """
    if mask is not None:
        # Ensure mask is boolean for indexing
        mask = mask.astype(bool)
        # Apply the mask
        gt = gt[mask]
        pred = pred[mask]

    # Calculate threshold-based accuracies
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    # Calculate RMSE
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    # Calculate RMSE log
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # Calculate absolute relative error
    abs_rel = np.mean(np.abs(gt - pred) / gt)

    mae = np.mean(np.abs(gt - pred))

    # Calculate squared relative error
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    # Calculate inverse root mean squared error
    inv_rmse = np.sqrt(np.mean(((1.0 / gt) - (1.0 / pred)) ** 2))

    # Calculate inverse mean absolute error
    inv_mae = np.mean(np.abs((1.0 / gt) - (1.0 / pred)))

    return abs_rel, sq_rel, rmse*1000, rmse_log ,inv_rmse*1000, inv_mae*1000, a1, a2, a3, mae*1000

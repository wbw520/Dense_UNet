from keras import backend as K

def sparse_focal(y_true,y_pred,gamma=2):
    alpha = [0.5,1,1,1,1]
    y_pred += K.epsilon()
    ce = -y_true * K.log(y_pred)
    weight = K.pow(1 - y_pred, gamma) * y_true
    fl = ce * weight * alpha
    out = K.mean(fl)
    return out

def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    """
    inse = K.sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = K.sum(output * output, axis=axis)
        r = K.sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = K.sum(output, axis=axis)
        r = K.sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    # old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    # new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = K.mean(dice)
    return 1-dice





import numpy as np
from prettytable import PrettyTable
import pystrum.pynd.ndutils as ndp

def vis(img1, img2):
    def autocontrast(img):
        return (255 / (img.max() - img.min())) * (img - img.min())

    result1 = np.zeros((img1.shape[0], img1.shape[1], 3))
    result2 = np.zeros((img1.shape[0], img1.shape[1], 3))

    img1[img1 > 10] += 30
    img1[img1 > 255] = 255
    img2[img2 > 10] += 30
    img2[img2 > 255] = 255

    result1[:, :, 0] = img1
    result1[:, :, 2] = img1
    result2[:, :, 1] = img2
    result2[:, :, 2] = img2
    result1 = autocontrast(result1)
    result2 = autocontrast(result2)
    return result1, result2, autocontrast(result1 + result2)

def dice(pred1, truth1):
    mask4_value1 = np.unique(pred1)
    mask4_value2 = np.unique(truth1)
    mask_value4 = list(set(mask4_value1) & set(mask4_value2))
    dice_list=[]
    for k in mask_value4[1:]:
        #print(k)
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        # print(intersection)
        dice_list.append(intersection / (np.sum(pred) + np.sum(truth)))
    return np.mean(dice_list)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 0)
    volshape = disp.shape[:-1]
    assert len(volshape) == 2, 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = ndp.volsize2ndgrid(volshape)
    # print(grid_lst)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    dfdx = J[0]
    dfdy = J[1]

    return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

def jacobian_determinant_3d(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    assert len(volshape) == 3, 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = ndp.volsize2ndgrid(volshape)
    # print(grid_lst)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    dx = J[0]
    dy = J[1]
    dz = J[2]

    # compute jacobian components
    Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
    Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
    Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

    return Jdet0 - Jdet1 + Jdet2
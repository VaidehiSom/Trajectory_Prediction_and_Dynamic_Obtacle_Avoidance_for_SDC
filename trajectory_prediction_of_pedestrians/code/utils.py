import pandas as pd
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#from sklearn.metrics import r2_score


def diff_axis_0(a):
    ret = a[1:] - a[:-1]
    ret = torch.cat([en_cuda(torch.zeros(1, a.size()[1])), ret])
    return ret


def en_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def create_rotation_matrices(angles, real=False):
    # Rotate
    if real:
        angle = ((np.pi / 2) - angles)
        return en_cuda(torch.Tensor([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]))
    matrices = []
    for i in range(angles.size()[0]):
        angle = ((np.pi / 2) - angles[i])
        matrices.append([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    return en_cuda(torch.Tensor(matrices))


def rotate(rotations, speeds, init=False):
    rotated_speeds = en_cuda(torch.zeros_like(speeds))
    if init:
        for i in range(speeds.size()[0]):
            rotated_speeds[i, :] = torch.matmul(rotations, speeds[i, :])
        return rotated_speeds.unsqueeze(0)

    for i in range(speeds.size()[0]):
        for j in range(speeds.size()[1]):
            rotated_speeds[i, j, :] = torch.matmul(
                rotations[j, :, :], speeds[i, j, :])
    return rotated_speeds


def load_data(scene, factor=10):
    '''Import Data file for training'''
    # Read data and group by frame
    selector = en_cuda(torch.LongTensor([2, 3]))
    df_positions = pd.read_table(scene, delim_whitespace=True, header=None)
    unique_frames = df_positions.iloc[:, 0].unique()
    unique_frames_to_index = dict(
        [(unique_frame[1], unique_frame[0]) for unique_frame in enumerate(unique_frames)])
    df_positions.loc[:, [2, 3]] = df_positions.loc[:, [2, 3]] * factor
    speeds = [None] * len(unique_frames)
    positions = [None] * len(unique_frames)
    rotations = [None] * len(unique_frames)
    first_positions = [None] * len(unique_frames)
    map_ped_index = {}
    # Group by ped_id and by starting frame
    index = 0
    df = []
    count = 0
    for _, x in df_positions.groupby(df_positions[1]):
        count += 1
        map_ped_index[x.iloc[0, 1]] = index
        index += 1
        mtrx = x.as_matrix()
        tmp = en_cuda(torch.Tensor(mtrx))
        df.append(mtrx)
        idx = unique_frames_to_index[x.iloc[0, 0]]
        copy_path = tmp.clone()
        copy_path[:, selector] = diff_axis_0(copy_path[:, selector])
        rotation = create_rotation_matrices(np.arctan2(
            copy_path[1, 3], copy_path[1, 2]), real=True)
        copy_path[:, selector] = rotate(rotation, copy_path[:, selector], True)
        if speeds[idx] is None:
            positions[idx] = []
            speeds[idx] = []
            rotations[idx] = []
            first_positions[idx] = []

        positions[idx].append(tmp.unsqueeze(0))
        speeds[idx].append(copy_path.unsqueeze(0))
        rotations[idx].append(rotation)
        first_positions[idx].append(x.iloc[8, :])
    # Get_first positions so that we can recover the position from the velocity

    positions_ret = []
    speeds_ret = []
    rotations_ret = []
    first_positions_ret = []
    max_length = 0
    for i in range(len(unique_frames)):
        if(positions[i] is not None):
            positions_ret.append(torch.cat(positions[i], 0).transpose(1, 0))
            speeds_ret.append(
                Variable(torch.cat(speeds[i], 0).transpose(1, 0)))
            tmp_rotations = []
            max_length = max(max_length, len(rotations[i]))
            for j in range(len(rotations[i])):
                tmp_rotations.append(torch.mul(rotations[i][j], en_cuda(
                    torch.Tensor([[1, -1], [-1, 1]]))).unsqueeze(0))
            rotations_ret.append(torch.cat(tmp_rotations, 0))
            first_positions_ret.append(
                en_cuda(torch.Tensor(first_positions[i])))

    return map_ped_index, speeds_ret, rotations_ret, positions_ret, first_positions_ret, count, df, max_length


def get_grid(hidden_states, frame_data, grid_size=4, max_dist=60):
    empty = Variable(en_cuda(torch.zeros(frame_data.size()[
                     0], grid_size * grid_size * hidden_states.size()[1])))
    # Check if there is only one pedestrian in the frame
    if frame_data.size()[0] <= 1:
        return empty
    # Create grid
    grid = Variable(en_cuda(torch.zeros(frame_data.size()[
                    0], grid_size, grid_size, hidden_states.size()[1])))
    increment = max_dist / (grid_size / 2)
    idx = list(range(frame_data.size()[0]))
    # Compute social grid for every pedestrian in the frame
    for i in range(frame_data.size()[0]):
        curr_ped = frame_data[i, :]
        idx_ = idx.copy()
        del idx_[i]
        # Compute distances to pedestrian i
        distances = frame_data[idx_, :].sub(curr_ped)
        # Compute others pedestrians poisitions in pedestrian i grid
        distances = torch.floor(
            torch.div(distances, increment)).add(int(grid_size / 2))
        # Filter the pedestrians which are far away
        selector = (distances[:, 0] < grid_size) & (distances[:, 0] > 0) & (
            distances[:, 1] < grid_size) & (distances[:, 1] > 0)
        selector = (selector).nonzero()
        # Check if there is pedestrian around pedestrian i which fits into square around pedestriand i
        if not len(selector.size()):
            continue
        distances = distances[(selector).squeeze(1)].type(
            torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)
        hidden_states_to_pool = hidden_states[(selector).squeeze(1)]
        idx_hidden = 0
        # Fill  pedestrian i grid
        for grid_idx in distances:
            grid[i, grid_idx[0], grid_idx[1], :] = torch.add(
                grid[i, grid_idx[0], grid_idx[1], :], hidden_states_to_pool[idx_hidden])
            idx_hidden += 1

    grid = grid.view(frame_data.size()[0], -1)
    return grid


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def get_coef(mux, muy, sx, sy, corr):
    # eq 20 -> 22 of Graves (2013)

    # The output must be exponentiated for the std devs
    o_sx = torch.exp(sx)
    o_sy = torch.exp(sy)
    # Tanh applied to keep it in the range [-1, 1]
    o_corr = torch.tanh(corr)
    return [mux, muy, o_sx, o_sy, o_corr]


def sample_gaussian_2d(mux, muy, sx, sy, rho):
    '''
    Function to sample a point from a given 2D normal distribution
    params:
    mux : mean of the distribution in x
    muy : mean of the distribution in y
    sx : std dev of the distribution in x
    sy : std dev of the distribution in y
    rho : Correlation factor of the distribution
    '''
    # Extract mean
    mean = [mux, muy]
    # Extract covariance matrix
    cov = [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]]
    # Sample a point from the multivariate normal distribution
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
    # PDF normal
    # TODO: How pi is computed ?? Is there any weighting
    normx = x.sub(mux)
    normy = y.sub(muy)
    # Calculate sx*sy
    sxsy = torch.mul(sx, sy)
    # Calculate the exponential factor
    z = (torch.div(normx, sx))**2 + (torch.div(normy, sy))**2 - 2 * \
        torch.div(torch.mul(rho, torch.mul(normx, normy)), sxsy)
    negRho = 1 - rho**2
    # exp part
    result = torch.exp(torch.div(-z, 2 * negRho))
    # Normalization constant
    denom = 2 * np.pi * torch.mul(sxsy, torch.sqrt(negRho))
    # Final PDF calculation
    result = torch.div(result, denom)
    # Check if the PDF was correctly computed
    #check = np.where(result.data.numpy() > 1)
    # if len(check[0]) > 0:
    #    print(x[check[0][0],check[1][0]].data[0],y[check[0][0],check[1][0]].data[0],mux[check[0][0],check[1][0]].data[0],muy[check[0][0],check[1][0]].data[0],sx[check[0][0],check[1][0]].data[0],sy[check[0][0],check[1][0]].data[0],rho[check[0][0],check[1][0]].data[0],result[check[0][0],check[1][0]].data[0])
    return result


def get_lossfunc(z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
    '''
    Function to calculate given a 2D distribution over x and y, and target data
    of observed x and y points
    params:
    z_mux : mean of the distribution in x
    z_muy : mean of the distribution in y
    z_sx : std dev of the distribution in x
    z_sy : std dev of the distribution in y
    z_rho : Correlation factor of the distribution
    x_data : target x points
    y_data : target y points
    '''
    # Calculate the PDF of the data w.r.t to the distribution
    result = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
    # TODO: WHY COMPUTING A MEAN ??
    # For numerical stability purposes
    epsilon = 1e-20

    # TODO: (resolve) I don't think we need this as we don't have the inner
    # summation
    # Apply the log operation
    result1 = -torch.log(torch.clamp(result, min=epsilon)
                         )  # Numerical stability

    # TODO: For now, implementing loss func over all time-steps
    # Sum up all log probabilities for each data point
    return torch.sum(result1).div(result1.size()[0])


def get_avg_displacement(true, pred):
    result = pred.sub(true)**2
    result = torch.sum(result, 2)
    norm = result.size()[0]
    result = torch.sum(result, 0)
    result = torch.div(result, norm)
    return result


def get_r_square(true, pred):
    predicted = pred**2
    true_ = true**2
    norm = true.size()[0] * true.size()[2]
    predicted = torch.sum(predicted, 1)
    true_ = torch.sum(true_, 1)
    result = torch.div(predicted, true_)

    result = torch.sum(result)
    result = result / norm
    return result


def plot_trajectory(true, pred, neighbs, n_paths, name, obs=8, xlim=None, ylim=None, debug=False):
    plt.figure()
    # Plot predicted and true trajectories
    true_ = true.cpu().numpy()
    x1, y1, x2, y2 = pred[:, 0].cpu().numpy(), pred[:, 1].cpu().numpy(), true_[
        :, 2], true_[:, 3]
    plt.plot(x1, y1, 'ro-', x2, y2, 'go-')
    plt.plot([x2[7], x1[0]], [y2[7], y1[0]], 'ro-')
    red_patch = mpatches.Patch(color='red', label='predicted trajectory')
    # Add timestamps
    for i in range(x1.shape[0]):
        plt.text(x1[i], y1[i], str(i))
        plt.text(x2[i + 8], y2[i + 8], str(i))

    map_ped_index = {}
    min_frame_all, step = true[0, 0], (true[1, 0] - true[0, 0])
    if step == 0:
        step = 1

    # Group neighbors by ped_id
    pedestrians = []
    idx = 0
    for frame in neighbs:
        for entry in frame:
            if entry[1] in map_ped_index:
                pedestrians[map_ped_index[entry[1]]].append(
                    np.expand_dims(entry.cpu().numpy(), axis=0))
            else:
                map_ped_index[entry[1]] = idx
                pedestrians.append([])
                pedestrians[idx].append(
                    np.expand_dims(entry.cpu().numpy(), axis=0))
                idx += 1

    if len(pedestrians):
        pedestrians = [np.concatenate(x, axis=0) for x in pedestrians]
        # Find closest neighbors
        dists = []
        for x in pedestrians:
            x = x[x[:, 0] != -1]
            min_frame = np.min(x[:, 0])
            max_frame = np.max(x[:, 0])
            sub_traj = true_[
                (true_[:, 0] >= min_frame) & (true_[:, 0] <= max_frame), :]
            dist = np.mean(
                np.sum((sub_traj[:, [2, 3]] - x[:, [2, 3]])**2, 1))
            dists.append(dist)

        idxs = np.argsort(np.array(dists))[:n_paths]
        pedestrians = [pedestrians[i] for i in idxs]
        # print(pedestrians[0])
        # Plot neighbors
        for path in pedestrians:
            path = path[path[:, 0] != -1]
            plt.plot(path[:, 2], path[:, 3], 'bo-')
            plt.text(path[0, 2], path[0, 3], int(
                (path[0, 0] - min_frame_all) / step))
            plt.text(path[-1, 2], path[-1, 3],
                     int((path[-1, 0] - min_frame_all) / step))

    green_patch = mpatches.Patch(color='green', label='true trajectory')
    blue_patch = mpatches.Patch(color='blue', label='neighboring trajectories')
    plt.legend(handles=[red_patch, green_patch, blue_patch])
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(name + '.png', dpi = 100)

    plt.close()
    return


def compute_speeds(a):
    ret = a[1:] - a[:-1]
    ret = torch.cat([en_cuda(torch.zeros(1, a.size()[1])), ret])
    return ret


def get_accuracy(true, pred):
    # Compute L2 distance and Final displacement error
    result = torch.sqrt(pred.sub(true)**2)
    result = torch.sum(result, 2)
    norm = result.size()[0]
    result_ = torch.sum(result, 0)
    result_ = torch.div(result_, norm)
    return result_, result[-1, :]


def get_grid_positions(frame_data, ped_id, ped_data=None, max_dist=30, grid_size=16, ped_id_=23):
    '''Get position of neighbors in the occupancy map with respect to the studied pedestrian'''
    # Check if there is no neighbors
    if (len(frame_data) == 0) or (len(frame_data.size()) == 0) or (frame_data.size()[0] == 0) or (ped_id_ == -1):
        return None, None

    increment = max_dist / (grid_size / 2)

    # Select the current pedestrian
    selector = None
    if ped_id is not None:
        selector = (frame_data[:, 1] == ped_id)
    if ped_data is not None:
        ped_dist = ped_data.clone().unsqueeze(0)
    else:
        ped_dist = frame_data[selector.nonzero().squeeze(1)]

    # Find all the other pedestrians in the frame
    if ped_id is None:
        others = frame_data.clone()
    else:
        others = frame_data[(~selector).nonzero().squeeze(1)]

    # Compute the other pedestrians distance to the pedestrian ped_id
    others[:, [2, 3]] = others[:, [2, 3]].sub(ped_dist)
    others[:, [2, 3]] = torch.floor(
        torch.div(others[:, [2, 3]], increment)).add(int(grid_size / 2))

    # Filter the pedestrians which are far away
    selector = (others[:, 2] < grid_size) & (others[:, 2] > 0) & (
        others[:, 3] < grid_size) & (others[:, 3] > 0)
    # Find position of invalid entries
    selector_unvalid = (others[:, 0] == -1)
    if not len((selector).nonzero().size()):
        return None, None

    # Indexes of valid neighbors + neighbors in grid
    returned_indexes = (~selector_unvalid & selector).nonzero()
    if len(returned_indexes.size()) == 0:
        return None, None
    returned_indexes = returned_indexes.squeeze(1)
    others = others[returned_indexes][:, [2, 3]].type(
        torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)

    return others, returned_indexes


def get_social_tensor(hiddens, frame_data=None, positions=None, grid_size=16):
    social_tensor = en_cuda(torch.zeros(
        grid_size, grid_size, hiddens[0].size()[2]))
    if positions is not None:
        idx = 0
        for coords in positions:
            social_tensor[coords[0], coords[1],
                          :] += hiddens[idx].data.squeeze(0).squeeze(0)
            idx += 1
        return social_tensor.view(1, -1)
    else:
        return social_tensor.view(1, -1)


def get_grid_with_pos(positions, grid_size=16):
    grid = en_cuda(torch.zeros(grid_size, grid_size))
    if positions is None:
        return grid.view(1, -1)
    # Fill the grid
    indices = torch.mul(positions[:, 0], grid_size).add(positions[:, 1])
    grid.put_(indices, en_cuda(torch.Tensor(
        [1])).expand_as(indices), accumulate=True)
    # Avg pooling on the grid
    #ret = F.avg_pool2d(Variable(grid).unsqueeze(0),8)
    grid = grid.view(1, -1)
    return grid
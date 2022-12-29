import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.autograd as autograd
from utils import *
use_cuda = False

def get_coef(mux, muy, sx, sy, corr):
    o_sx = torch.exp(sx)
    o_sy = torch.exp(sy)
    
    o_corr = torch.tanh(corr)
    return [mux, muy, o_sx, o_sy, o_corr]

def en_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor

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

class VanillaLSTMNet(nn.Module):
    def __init__(self):

        super(VanillaLSTMNet, self).__init__()
        
        # Inputs to the LSTMCell's are (input, (h_0, c_0)):
        # 1. input of shape (batch, input_size): tensor containing input 
        # features
        # 2a. h_0 of shape (batch, hidden_size): tensor containing the 
        # initial hidden state for each element in the batch.
        # 2b. c_0 of shape (batch, hidden_size): tensor containing the 
        # initial cell state for each element in the batch.
        
        # Outputs: h_1, c_1
        # 1. h_1 of shape (batch, hidden_size): tensor containing the next 
        # hidden state for each element in the batch
        # 2. c_1 of shape (batch, hidden_size): tensor containing the next 
        # cell state for each element in the batch
        
        # set parameters for network architecture
        self.embedding_size = 64
        self.input_size = 2
        self.output_size = 2
        self.dropout_prob = 0.5 
        
        # linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        
        # define lstm cell
        self.lstm_cell = nn.LSTMCell(self.embedding_size, self.embedding_size)

        # linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.embedding_size, self.output_size)
        
        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        
        pass
 
    def forward(self, observed_batch, pred_len = 0):
        
        '''
        Forward pass for the model.
        '''
        
        output_seq = []

        ht = torch.zeros(observed_batch.size(1), self.embedding_size, dtype=torch.float)
        ct = torch.zeros(observed_batch.size(1), self.embedding_size, dtype=torch.float)

        seq, peds, coords = observed_batch.shape
        #Feeding the observed trajectory to the network
        for step in range(seq):
            observed_step = observed_batch[step, :, :]
            lin_out = self.input_embedding_layer(observed_step.view(peds,2))
            ht, ct = self.lstm_cell(lin_out, (ht, ct))
            out = self.output_layer(self.dropout(ht))

        print("out's shape:", out.shape)
        #Getting the predicted trajectory from the pedestrian 
        for i in range(pred_len):
            lin_out = self.input_embedding_layer(out)
            ht, ct = self.lstm_cell(lin_out, (ht,ct))
            out = self.output_layer(self.dropout(ht))
            output_seq += [out]

        output_seq = torch.stack(output_seq).squeeze() # convert list to tensor
        return output_seq


class GRUNet(nn.Module):
    def __init__(self):
        super(GRUNet, self).__init__()
        
        ''' Inputs to the GRUCell's are (input, (h_0, c_0)):
         1. input of shape (batch, input_size): tensor containing input 
         features
         2a. h_0 of shape (batch, hidden_size): tensor containing the 
         initial hidden state for each element in the batch.
         2b. c_0 of shape (batch, hidden_size): tensor containing the 
         initial cell state for each element in the batch.
        
         Outputs: h_1, c_1
         1. h_1 of shape (batch, hidden_size): tensor containing the next 
         hidden state for each element in the batch
         2. c_1 of shape (batch, hidden_size): tensor containing the next 
         cell state for each element in the batch '''
        
        # set parameters for network architecture
        self.embedding_size = 64
        self.rnn_size=128
        self.input_size = 2
        self.output_size = 2
        self.dropout_prob = 0.5
        if(use_cuda):
            self.device = torch.device("cuda:0") # to run on GPU
        else:
            self.device=torch.device("cpu")

        # linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        
        # define gru cell
        self.gru_cell = nn.GRUCell(self.embedding_size, self.rnn_size)

        # linear layer to map the hidden state of gru to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)
        
        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        
        pass
 
    def forward(self, observed_batch, pred_len = 0):
        ''' this function takes the input sequence and predicts the output sequence. 
        
            args:
                observed_batch (torch.Tensor) : input batch with shape <seq length x num pedestrians x number of dimensions>
                pred_len (int) : length of the sequence to be predicted.

        '''
        output_seq = []

        ht = torch.zeros(observed_batch.size(1), self.rnn_size,device=self.device, dtype=torch.float)
        ct = torch.zeros(observed_batch.size(1), self.rnn_size,device=self.device, dtype=torch.float)
        seq, peds, coords = observed_batch.shape

        # feeding the observed trajectory to the network
        for step in range(seq):
            observed_step = observed_batch[step, :, :]
            lin_out = self.input_embedding_layer(observed_step.view(peds,2))
            input_embedded=self.dropout(self.relu(lin_out))
            ht = self.gru_cell(input_embedded, ht)
            out = self.output_layer(ht)

        # getting the predicted trajectory from the pedestrian 
        for i in range(pred_len):
            lin_out = self.input_embedding_layer(out)
            input_embedded=self.dropout(self.relu(lin_out))
            ht= self.gru_cell(input_embedded, ht)
            out = self.output_layer(ht)
            output_seq += [out]
            
        output_seq = torch.stack(output_seq).squeeze() # convert list to tensor
        return output_seq


class VLSTM(nn.Module):
    '''
    Class representing the Social LSTM model
    '''

    def __init__(self, args):
        '''
        Initializer function
        params:
        args: Training arguments
        '''
        super(VLSTM, self).__init__()
        self.embedded_input = args['embedded_input']
        self.hidden_size = args['hidden_size']
        self.embedding = nn.Linear(2, self.embedded_input)
        self.lstm = nn.LSTM(self.embedded_input, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, 5)

    def forward(self, x_data, num=12):
        # Embedd input
        embedded_input = [F.relu(self.embedding(x_data[i, :, :])).unsqueeze(
            0) for i in range(x_data.size()[0])]
        embedded_input = torch.cat(embedded_input, 0)
        inputs = embedded_input[:-1, :, :]

        # Feed embedded in to lstm
        hidden = (Variable(en_cuda(torch.randn(1, inputs.size()[1], self.hidden_size))), Variable(
            en_cuda(torch.randn((1, inputs.size()[1], self.hidden_size)))))  # clean out hidden state
        out, hidden = self.lstm(inputs, hidden)
        last = embedded_input[-1, :, :].unsqueeze(0)

        results = []
        # Generate outputs
        speeds = []
        for i in range(num):
            out, hidden = self.lstm(last, hidden)
            linear_out = self.output(out.squeeze(0))

            linear_out = linear_out.split(1, 1)
            res_params = get_coef(*linear_out)
            results.append(torch.cat(res_params, 1).unsqueeze(0))
            res_params = torch.cat(res_params, 1)

            # Iterate over all batch points
            last = []
            temp_points = []
            for idx in range(res_params.size()[0]):
                mux = res_params.data[idx, 0]
                muy = res_params.data[idx, 1]
                sx = res_params.data[idx, 2]
                sy = res_params.data[idx, 3]
                rho = res_params.data[idx, 4]
                speed = en_cuda(torch.Tensor(
                    sample_gaussian_2d(mux, muy, sx, sy, rho)))
                last.append(speed.unsqueeze(0))
            last = torch.cat(last, 0)
            speeds.append(last.unsqueeze(0))
            last = Variable(last)
            last = F.relu(self.embedding(last)).unsqueeze(0)
        results = torch.cat(results, 0)
        speeds = (torch.cat(speeds, 0))
        return results, speeds

    def get_hidden_states(self, input_, hidden):
        pos = F.relu(self.embedding(input_)).unsqueeze(0)
        out, hidden = self.lstm(pos, hidden)
        return hidden


class SLSTM(nn.Module):
    '''
    Class representing the Social LSTM model
    '''

    def __init__(self, args):
        '''
        Initializer function
        params:
        args: Training arguments
        '''
        super(SLSTM, self).__init__()
        self.embedded_input = args['embedded_input']
        self.embedding_occupancy_map = args['embedding_occupancy_map']
        self.use_speeds = args['use_speeds']
        self.grid_size = args['grid_size']
        self.max_dist = args['max_dist']
        self.hidden_size = args['hidden_size']
        model_checkpoint = args['trained_model']
        self.trained_model = VLSTM(args)
        if model_checkpoint is not None:
            if torch.cuda.is_available():
                load_params = torch.load(model_checkpoint)
            else:
                load_params = torch.load(
                    model_checkpoint, map_location=lambda storage, loc: storage)
            self.trained_model.load_state_dict(load_params['state_dict'])

            # Freeze model
            for param in self.trained_model.parameters():
                param.requires_grad = False

        self.embedding_spatial = nn.Linear(2, self.embedded_input)
        self.embedding_o_map = nn.Linear(
            (args['grid_size']**2) * self.hidden_size,
            self.embedding_occupancy_map)
        self.lstm = nn.LSTM(self.embedded_input +
                            self.embedding_occupancy_map, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, 5)

    def forward(self, input_data, grids, neighbors, first_positions, num=12):
        selector = en_cuda(torch.LongTensor([2, 3]))
        #grids = Variable(grids)
        obs = 8
        # Embedd input and occupancy maps
        # Iterate over frames
        hiddens_neighb = [(autograd.Variable(en_cuda(torch.zeros(1, 1, self.hidden_size))), autograd.Variable(
            en_cuda(torch.zeros((1, 1, self.hidden_size))))) for i in range(len(neighbors[0]) + 1)]
        inputs = None
        temp = None
        for i in range(input_data.size()[0]):
            temp = input_data[i, :].unsqueeze(0)
            embedded_input = F.relu(self.embedding_spatial(
                temp[:, selector])).unsqueeze(0)
            # Iterate over peds if there is neighbors
            social_tensor = None

            # Check if the pedestrians has neighbors
            if len(neighbors[0]):
                all_frame = torch.cat(
                    [input_data[i, :].data.unsqueeze(0), neighbors[i]], 0)
                valid_indexes_center = grids[i][1]
                # Iterate over valid neighbors if exists

                if valid_indexes_center is not None:
                    buff_hidden_c,buff_hidden_h = [],[]
                    for k in valid_indexes_center:
                        # Get hidden states from VLSTM
                        buff_hidden_c.append(hiddens_neighb[k+1][0])
                        buff_hidden_h.append(hiddens_neighb[k+1][1])

                    # Compute neighbor speed
                    if i>0:
                        last_obs = neighbors[i - 1][valid_indexes_center, :]
                        vlstm_in = neighbors[i][valid_indexes_center, :][:,[
                                    2, 3]] - last_obs[:,[2, 3]]
                        select = (last_obs[:,1] == -1).nonzero()
                        if len(select.size()):
                            vlstm_in[select.squeeze(1)] = 0

                    else:
                        vlstm_in = en_cuda(torch.zeros(len(valid_indexes_center),2))



                    hiddens_tmp = self.trained_model.get_hidden_states(
                        Variable(vlstm_in), (torch.cat(buff_hidden_c,1),torch.cat(buff_hidden_h,1)))

                    for idx,k in enumerate(valid_indexes_center):
                        hiddens_neighb[k+1] = (hiddens_tmp[0][:,idx,:].unsqueeze(0),hiddens_tmp[1][:,idx,:].unsqueeze(0))

                    hidden_relu = [F.relu(hiddens_neighb[e + 1][0])
                                   for e in valid_indexes_center]

                    social_tensor = Variable(get_social_tensor(
                        hidden_relu, positions=grids[i][0], grid_size=self.grid_size))
                else:
                    social_tensor = Variable(en_cuda(torch.zeros(
                        1, (self.grid_size**2) * self.hidden_size)))
            else:
                social_tensor = Variable(en_cuda(torch.zeros(
                    1, (self.grid_size**2) * self.hidden_size)))

            embedded_o_map = F.relu(
                self.embedding_o_map(social_tensor)).unsqueeze(0)
            inputs = torch.cat([embedded_input, embedded_o_map], 2)
            if i == (obs - 1):
                break
            hiddens_to_feed = hiddens_neighb[0]
            self.lstm.flatten_parameters()
            out, hidden = self.lstm(inputs, hiddens_to_feed)
            hiddens_neighb[0] = hidden

        # Predict

        last = inputs

        results = []
        # Generate outputs
        points = []
        first_positions_c = first_positions.clone()
        for i in range(num):
            # get gaussian params for every point in batch
            hiddens_to_feed = hiddens_neighb[0]
            self.lstm.flatten_parameters()
            out, hidden = self.lstm(inputs, hiddens_to_feed)
            hiddens_neighb[0] = hidden

            linear_out = self.output(out.squeeze(0))

            linear_out = linear_out.split(1, 1)
            res_params = get_coef(*linear_out)
            results.append(torch.cat(res_params, 1).unsqueeze(0))
            res_params = torch.cat(res_params, 1)

            last_speeds = []
            last_grids = []
            temp_points = []
            mux = res_params.data[0, 0]
            muy = res_params.data[0, 1]
            sx = res_params.data[0, 2]
            sy = res_params.data[0, 3]
            rho = res_params.data[0, 4]
            # Sample speeds
            speed = en_cuda(torch.Tensor(
                sample_gaussian_2d(mux, muy, sx, sy, rho)))
            pts = torch.add(speed, first_positions_c[0, :])
            first_positions_c[0, :] = pts

            # Compute embeddings  and social grid

            last_speeds = speed.unsqueeze(0)
            last_speeds = Variable(last_speeds)
            pts_frame = pts.unsqueeze(0)

            # SOCIAL TENSOR
            # Check if neighbors exists
            if(len(neighbors[i + obs])):
                pts_w_metadata = en_cuda(torch.Tensor(
                    [[neighbors[i + obs][0, 0], input_data[0, 1].data[0], pts[0], pts[1]]]))
                frame_all = torch.cat(
                    [pts_w_metadata, neighbors[i + obs]], 0)
                # Get positions in social_grid
                (indexes_in_grid, valid_indexes) = get_grid_positions(neighbors[i + obs], None, ped_data=pts_frame.squeeze(0),
                                                                      grid_size=self.grid_size, max_dist=self.max_dist)
                if(valid_indexes is not None):

                    buff_hidden_c,buff_hidden_h = [],[]
                    for k in valid_indexes:
                        # Get hidden states from OLSTM
                        buff_hidden_c.append(hiddens_neighb[k+1][0])
                        buff_hidden_h.append(hiddens_neighb[k+1][1])

                    # Compute neighbor speed
                    last_obs = neighbors[i + obs - 1][valid_indexes, :]
                    vlstm_in = neighbors[i+ obs][valid_indexes, :][:,[
                                2, 3]] - last_obs[:,[2, 3]]
                    select = (last_obs[:,1] == -1).nonzero()
                    if len(select.size()):
                        vlstm_in[select.squeeze(1)] = 0


                    hiddens_tmp = self.trained_model.get_hidden_states(
                        Variable(vlstm_in), (torch.cat(buff_hidden_c,1),torch.cat(buff_hidden_h,1)))

                    for idx,k in enumerate(valid_indexes):
                        hiddens_neighb[k+1] = (hiddens_tmp[0][:,idx,:].unsqueeze(0),hiddens_tmp[1][:,idx,:].unsqueeze(0))

                    # Compute social tensor
                    hidden_relu = [F.relu(hiddens_neighb[e + 1][0])
                                   for e in valid_indexes]
                    social_tensor = Variable(get_social_tensor(
                        hidden_relu, positions=indexes_in_grid, grid_size=self.grid_size))
                else:
                    social_tensor = Variable(en_cuda(torch.zeros(
                        1, (self.grid_size**2) * self.hidden_size)))
            else:
                social_tensor = Variable(en_cuda(torch.zeros(
                    1, (self.grid_size**2) * self.hidden_size)))

            last_speeds = F.relu(
                self.embedding_spatial(last_speeds)).unsqueeze(0)
            last_grids = F.relu(self.embedding_o_map(
                social_tensor)).unsqueeze(0)
            last = torch.cat([last_speeds, last_grids], 2)
            points.append(pts_frame.unsqueeze(0))

        results = torch.cat(results, 0)
        points = torch.cat(points, 0)
        return results, points


class OLSTM(nn.Module):
    '''
    Class representing the Social LSTM model
    '''

    def __init__(self, args):
        '''
        Initializer function
        params:
        args: Training arguments
        '''
        super(OLSTM, self).__init__()
        self.embedded_input = args['embedded_input']
        self.embedding_occupancy_map = args['embedding_occupancy_map']
        self.use_speeds = args['use_speeds']
        self.grid_size = args['grid_size']
        self.max_dist = args['max_dist']
        self.hidden_size = args['hidden_size']
        self.embedding_spatial = nn.Linear(2, self.embedded_input)
        self.embedding_o_map = nn.Linear(args['grid_size']**2, self.embedding_occupancy_map)
        self.lstm = nn.LSTM(self.embedded_input +
                            self.embedding_occupancy_map, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, 5)

    def forward(self, input_data, grids, neighbors,first_positions, num=12):

        selector = en_cuda(torch.LongTensor([2, 3]))
        embedded_input = []
        grids = Variable(grids)
        obs = 8
        # Embedd input and occupancy maps
        for i in range(input_data.size()[0]):
            temp = input_data[i, :, :]
            embedded_input.append(
                F.relu(self.embedding_spatial(temp[:, selector])).unsqueeze(0))

        embedded_input = torch.cat(embedded_input, 0)
        embedded_o_map = [F.relu(self.embedding_o_map(grids[i, :, :])).unsqueeze(
            0) for i in range(grids.size()[0])]

        embedded_o_map = torch.cat(embedded_o_map, 0)
        inputs = torch.cat([embedded_input, embedded_o_map], 2)

        # Feed embedded in to lstm
        hidden = (autograd.Variable(en_cuda(torch.randn(1, inputs.size()[1], self.hidden_size))), autograd.Variable(
            en_cuda(torch.randn((1, inputs.size()[1], self.hidden_size)))))  # clean out hidden state
        self.lstm.flatten_parameters()
        out, hidden = self.lstm(inputs[:-1, :, :], hidden)

        last = inputs[-1, :, :].unsqueeze(0)

        results = []
        # Generate outputs
        points = []
        position_tracker = None
        if self.use_speeds:
            position_tracker = first_positions.clone()

        for i in range(num):
            # get gaussian params for every point in batch
            self.lstm.flatten_parameters()
            out, hidden = self.lstm(last, hidden)

            linear_out = self.output(out.squeeze(0))

            linear_out = linear_out.split(1, 1)
            res_params = get_coef(*linear_out)
            results.append(torch.cat(res_params, 1).unsqueeze(0))
            res_params = torch.cat(res_params, 1)

            # Iterate over all batch points
            last_speeds = []
            last_grids = []
            temp_points = []
            for idx in range(res_params.size()[0]):

                mux = res_params.data[idx, 0]
                muy = res_params.data[idx, 1]
                sx = res_params.data[idx, 2]
                sy = res_params.data[idx, 3]
                rho = res_params.data[idx, 4]
                # Sample speeds
                speed = en_cuda(torch.Tensor(
                    sample_gaussian_2d(mux, muy, sx, sy, rho)))
                if self.use_speeds:
                    pts = torch.add(speed, position_tracker[idx, :])
                    position_tracker[idx, :] = pts.clone()
                else:
                    pts = speed
                temp_points.append(pts.unsqueeze(0))
                # Compute current position and get the occupancy map
                grid = Variable(get_grid(neighbors[idx][i+obs], None, ped_data= pts, max_dist=self.max_dist, grid_size=self.grid_size))
                last_speeds.append(speed.unsqueeze(0))
                last_grids.append(grid)

            # Compute embeddings
            last_speeds = torch.cat(last_speeds, 0)
            last_speeds = Variable(last_speeds)
            last_grids = torch.cat(last_grids, 0)
            last_speeds = F.relu(
                self.embedding_spatial(last_speeds)).unsqueeze(0)
            last_grids = F.relu(self.embedding_o_map(last_grids)).unsqueeze(0)
            last = torch.cat([last_speeds, last_grids], 2)
            points.append(torch.cat(temp_points, 0).unsqueeze(0))

        results = torch.cat(results, 0)
        points = torch.cat(points, 0)
        return results, points

    def get_hidden_states(self,input_,grid,hidden):
        pos = F.relu(
            self.embedding_spatial(input_)).unsqueeze(0)
        grids = F.relu(self.embedding_o_map(grid)).unsqueeze(0)
        lstm_input = torch.cat([pos, grids], 2)
        out, hidden = self.lstm(lstm_input, hidden)
        return hidden
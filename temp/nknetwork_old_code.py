
#Autor Pugavkomm
#version 1.4_beta

import numpy as np
import torch
from tqdm import tqdm
import traceback
DICTIONARY_TYPE_CONNECTION =   {
                                'current_tanh': 0, 
                                'voltage_tanh': 1,
                                'current_exponential': 2, 
                                'voltage_exponential': 3,
                                }
DICTIONARY_TYPE_METHOD =  {
                            'FORCE': 0,
                            'fFORCE': 1
                            }
#print('current_tan' in DICTIONARY_TYPE_CONNECTION)

def PCA(data, k=3, mode='cpu'):
    '''This function returns the 
    main components from the data. 
    The number of returned main 
    components is specified by 
    the second argument (default: k = 3).'''

    
    if mode == 'cpu':
        X = torch.from_numpy(data)
        X_mean = torch.mean(X,0)
        X = X - X_mean.expand_as(X)
        U,S,V = torch.svd(torch.t(X))
        return torch.mm(X,U[:,:k])
    elif mode == 'gpu':
        X = torch.cuda.FloatTensor(data)
        X_mean = torch.mean(X,0)
        X = X - X_mean.expand_as(X)
        U,S,V = torch.svd(torch.t(X))
        return torch.mm(X,U[:,:k])
class Network_K_N:
    '''A data type describing a network. s
    It stores its parameters and can generate 
    a weight matrix.'''
    numbers_parameters = 5
    min_v = 0   # optimal
    max_v = 0.6 # optiamal
    

    def __init__(self, N, p_neurons):
        # p_neurons = [eps, a, d, beta, J]
        self.__text_error_arg = 'Invalid number of network arguments'
        self.__N = N 
        self.__w_in = torch.cuda.FloatTensor(N, 1).zero_()
        self.__stop_feedback = [] # default: 0, e.t. fedback always 
        self.f_in = [0]
        self.__connection_parameters = []
        try:
            if len(p_neurons) != self.__class__.numbers_parameters:
                raise Exception(self.__text_error_arg)
        except Exception as err:
            print('Error:\n', traceback.format_exc())
        self.__p_neurons = p_neurons
        # parameters
        #self.__eps   = p_neurons[0]
        #self.__a     = p_neurons[1]
        #self.__d     = p_neurons[2]
        #self.__beta  = p_neurons[3]
        #self.__J     = p_neurons[4]

    def one_neuron(self, x0, y0, iteration):
        '''The dynamics of one neuron. Return x, y.'''
        x, y = np.zeros(iteration), np.zeros(iteration)
        x[0], y[0] = x0, y0
        for i in range(iteration - 1):
            x[i + 1] = x[i] + x[i] * (x[i] - self.__p_neurons[1]) *\
                (1 - x[i]) - self.__p_neurons[3] *\
                (x[i] > self.__p_neurons[2]) - y[i]
            y[i + 1] = y[i] + self.__p_neurons[0] *\
                (x[i] - self.__p_neurons[4])
        return x, y 


    @property
    def connection_parameters(self):
        return self.__connection_parameters
    

    @connection_parameters.setter
    def connection_parameters(self, new_value):
        self.__connection_parameters = new_value


    @property
    def stopping_feedback(self):
        return self.__stop_feedback
    
    @stopping_feedback.setter
    def stopping_feedback(self, value):
        self.__stop_feedback = value


    def init_input_weight(self, dim_input, mean=0, dev=1, type_generate='uniform', **kwargs):
        '''Create the input weights'''
        if type_generate == 'uniform':
            self.__w_in = torch.cuda.FloatTensor(self.__N, dim_input).uniform_(mean - dev, mean + dev)
        elif type_generate == 'normal':
            self.__w_in = torch.cuda.FloatTensor(self.__N, dim_input).normal_(mean, dev)


    def init_weight(self, N, G, p, method = 'default'):
        '''Returns a (NxN) matrix of weights between network nodes.\n
        Parameters:\n
        N - network size,\n
        G is the strength of the bonds,\n
        p is the sparsity of the matrix,\n
        method - a generation method. 
        By default, <default> is the normal distribution.'''
        if method == 'default': 
            OMEGA = np.random.normal(0, 1, (N, N)) 
            OMEGA *= G * (np.random.random([N, N]) < p)
            OMEGA /= (np.sqrt(N) * p)
        return OMEGA


    def input_function(self, f_in):
        self.f_in = f_in

    @property
    def input_weights(self):
        return self.__w_in


    @input_weights.setter
    def input_weights(self, w_in):
        self.__w_in = w_in

    @property
    def size(self):
        return self.__N


    @size.setter
    def size(self, new_size):
        self.__N = new_size


    @property
    def parameters_neurons(self):
        '''If there are no arguments, it returns the 
        network parameters in the form of a list [eps, a, d, beta, J]. 
        If the list is given as an argument [eps, a, d, beta, J], 
        then the parameters of the neuron are replaced by the input ones.'''
        return self.__p_neurons


    @parameters_neurons.setter
    def parameters_neurons(self, p_neurons):
        try:
            if len(p_neurons) != self.__class__.numbers_parameters:
                raise Exception(self.__text_error_arg)
        except Exception as err:
            print('Error:\n', traceback.format_exc())
        self.__p_neurons = p_neurons


def current_tanh(I, v, JX, p_neurons, vpeak, i,  w_in, f_in = torch.cuda.FloatTensor(1, 1).zero_()):
    Is = I
    
    I_out = I + p_neurons[0] * (v - p_neurons[4] - JX - torch.mm(w_in, f_in))
    v_out =  v + v * (v - p_neurons[1]) * (1 - v) - p_neurons[3] * ((v > p_neurons[2]).float()) - Is 
    r_out = torch.tanh(v_out) * (v_out > vpeak).float()
    return I_out, v_out, r_out

def voltage_tanh(I, v, JX, p_neurons, vpeak, i,  w_in, f_in = torch.cuda.FloatTensor(1, 1).zero_()):
    Is = I
    
    I_out = I + p_neurons[0] * (v - p_neurons[4] )
    v_out =  v + v * (v - p_neurons[1]) * (1 - v) - p_neurons[3] * ((v > p_neurons[2]).float()) - Is +  JX + torch.mm(w_in, f_in)
    r_out = torch.tanh(v_out) * (v_out > vpeak).float()
    return I_out, v_out, r_out


def voltage_exponential(I, v, r, hr, JX, M1, M2, p_neurons, vpeak, i, w_in, f_in = torch.cuda.FloatTensor(1, 1).zero_()):
    Is = I
    I_out = I + p_neurons[0] * (v - p_neurons[4] )
    v_out =  v + v * (v - p_neurons[1]) * (1 - v) - p_neurons[3] * ((v > p_neurons[2]).float()) - Is + JX + torch.mm(w_in, f_in)
    r_out = r * (1 - 1 / M1) + hr
    hr_out = hr * (1 - 1/M2) + (v >= vpeak).float() / M1 / M2
    return I_out, v_out, r_out, hr_out

#def voltage_tanh(I, v, JX, p_neurons, vpeak, i,  w_in = 0, f_in = 0):
#    Is = I
#    I_out = I + p_neurons[0] * (v - p_neurons[4] )
#    v_out =  v + v * (v - p_neurons[1]) * (1 - v) - p_neurons[3] * ((v > p_neurons[2]).float()) \
#        - Is + JX
#    r_out = torch.tanh(v_out) * (v_out > vpeak).float()
#    return I_out, v_out, r_out

#.......................
#def current_exponential
#.......................

#def force_iteration(Pinv, cd, r, error, phi):
##    Pinv_out = Pinv - torch.mm(cd, torch.t(cd))/(1.0 + torch.mm(torch.t(r), cd))
#    cd_out = torch.mm(Pinv_out, r)
    #print(error)
#    phi_out = phi - torch.mm(cd_out, torch.t(error))
##    return Pinv_out, cd_out, phi_out


def force_iteration(Pinv, cd, r, error, phi):
    cd_out = torch.mm(Pinv, r)
    Pinv_out = Pinv - torch.mm(cd_out, torch.t(cd_out))/(1.0 + torch.mm(torch.t(r), cd_out))
    cd_out = torch.mm(Pinv_out, r)
    #print(error)
    phi_out = phi - torch.mm(cd_out, torch.t(error))
    return Pinv_out, cd_out, phi_out

def force_learn(connection, network, p_method, imin, icrit, nt, f_out,
dimteacher = 1, record = 0):
    try:
        if len(p_method) != 5:
            Exception('Invalid number of algorithm arguments')
    except Exception as err:
        print('Error:\n', traceback.format_exc())
    
    f_in = network.f_in
    if len(f_in) == 1:
        f_in = np.zeros(nt)
    
    if connection == 'voltage_exponential' or connection == 'current_exponential':
        try:
            if len(network.connection_parameters) != 2:
                raise Exception('Two parameters must be specified for this type of communication')
        except Exception as err:
            print('Error:\n', traceback.format_exc())
        M1, M2 = network.connection_parameters
    G, Q, p, speed, vpeak = p_method
    p_neurons = network.parameters_neurons
    eps, a, d, beta, J = p_neurons
    N = network.size
    OMEGA1 = network.init_weight(N, G, p)
    w_in = network.input_weights
    prelearning = np.zeros((N, record))
    postlearning = np.zeros((N, record))
    Iprelearning = np.zeros((N, record))
    Ipostlearning = np.zeros((N, record))
    OMEGA = torch.cuda.FloatTensor(OMEGA1)
    E = torch.cuda.FloatTensor(N, dimteacher).uniform_(-1, 1) * Q
    v = torch.cuda.FloatTensor(N, 1).uniform_(Network_K_N.min_v, Network_K_N.max_v) # test
    I = torch.cuda.FloatTensor(N, 1).zero_()
    r = torch.cuda.FloatTensor(N, 1).zero_()
    hr = torch.cuda.FloatTensor(N, 1).zero_()
    ISPC = torch.cuda.FloatTensor(N, 1).zero_()
    JD = torch.cuda.FloatTensor(N, 1).zero_()
    cd = torch.cuda.FloatTensor(N, 1).zero_()
    phi = torch.cuda.FloatTensor(N, dimteacher).zero_()
    JX = torch.cuda.FloatTensor(N, 1).zero_()
    Pinv = torch.cuda.FloatTensor(np.eye(N) * speed) 
    zout = torch.cuda.FloatTensor(dimteacher, 1).zero_() # out network
    Is = torch.cuda.FloatTensor(N, 1)                    # bufer for current\
    ##error = torch.cuda.FloatTensor(dimteacher, 1).zero_()
    vsave = np.zeros(nt)
    isave = np.zeros(nt)
    ersave = np.zeros((dimteacher, nt))
    rsave = np.zeros((N, record))
    rsave_post = np.zeros((N, record))
    Jsave = np.zeros((N, record))
    Jsave_post = np.zeros((N, record))
    stop_feedback = network.stopping_feedback
    fout_gpu = torch.reshape(torch.cuda.FloatTensor(f_out), (nt, dimteacher, 1))
    error_gpu = torch.cuda.FloatTensor(nt, dimteacher, 1).zero_()
    if len(stop_feedback) == 0:
        stop_feedback = np.zeros(nt)
    phi_place_holder = torch.cuda.FloatTensor(dimteacher, N).zero_()
    for i in tqdm(range(nt)):
        JX =  torch.mm(E, zout) + ISPC
        if connection == 'current_tanh':
            I, v, r = current_tanh(I, v, JX, p_neurons, vpeak, i, w_in, torch.cuda.FloatTensor([[f_in[i]]]))
        elif connection == 'voltage_exponential':
            I, v, r, hr = voltage_exponential(I, v, r, hr, JX, M1, M2, p_neurons, vpeak, i, w_in, torch.cuda.FloatTensor([[f_in[i]]]))
        elif connection == 'voltage_tanh':
            I, v, r = voltage_tanh(I, v, JX, p_neurons, vpeak, i, w_in, torch.cuda.FloatTensor([[f_in[i]]]))
        ISPC = torch.mm(OMEGA, r)
        if not stop_feedback[i]:
            zout = torch.mm(torch.t(phi), r)
            error_gpu[i] = zout -  fout_gpu[i]
            #print(fin_gpu[i])
        else:
            
            zout = torch.cuda.FloatTensor(1, 1).zero_()
            error_gpu[i] = torch.cuda.FloatTensor(1, 1).zero_()
        #ersave[:, i] = error.cpu().numpy()[:, 0]
        # Training 
        if imin < i < icrit and not stop_feedback[i]:
            Pinv, cd, phi = force_iteration(Pinv, cd, r, error_gpu[i], phi)
        # Record data
        if (i < record):
            prelearning[:, i] = v.cpu()[:, 0]
            Iprelearning[:, i] = I.cpu()[:, 0]
            rsave[:, i] = r.cpu()[:,0]
            Jsave[:, i] = (p_neurons[4]  + JX).cpu()[:, 0]
        elif (i > icrit and i < icrit + record):
            Ipostlearning[:, i-icrit] = I.cpu()[:, 0]
            postlearning[:, i - icrit] = v.cpu()[:, 0]
            Jsave_post[:, i - icrit] = (p_neurons[4] + JX).cpu()[:, 0]
            rsave_post[:, i - icrit] =  r.cpu()[:,0]
    ersave[:, :] = torch.t(error_gpu[:, :, 0]).cpu().numpy()
    print('shape = {}'.format(torch.t(error_gpu[:, :, 0]).cpu().numpy().shape))
    return ersave, prelearning, postlearning, Iprelearning, Ipostlearning, rsave, rsave_post, Jsave, Jsave_post, phi


def create_win(size_network, dim_input, type_generate = 'uniform', median=0, dev=1):
    if type_generate == 'uniform':
        w_in = torch.cuda.FloatTensor(size_network, dim_input).uniform_(median - dev, median + dev)
    elif type_generate == 'normal':
        w_in = torch.cuda.FloatTensor(size_network, dim_input).normal_(median, dev)
    return w_in



def learn(method, connection, network, p_method, imin, icrit, nt, record, f_out, dimteacher=1):
    try:
        if connection not in DICTIONARY_TYPE_CONNECTION:
            text_available = '\navailable types:\n'
            for i in DICTIONARY_TYPE_CONNECTION:
                text_available += '{}{}\n'.format(i.ljust(20, '.'), DICTIONARY_TYPE_CONNECTION[i])
                
            raise Exception('Invalid connection: ' + connection + text_available)
    except Exception as err:
        print('Error:\n', traceback.format_exc())
    else:
        if method == 'FORCE':
            return force_learn(connection, network, p_method, imin, icrit, nt, 
                                f_out, record = record, dimteacher=dimteacher)

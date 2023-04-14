import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# This file trains the system of neural networks on combined time-series data to
# test how well it generalizes.


import numpy as np
import itertools
import training_function as train
import accessory_functions as acc_f
import json


# Generates all possible combinations of data sets one wants to train the NNs on
def generate_combs(min_size,num_data):
    comb_set = []
    datasets = range(num_data)
    for l in range(min_size,num_data+1):
        for subset in itertools.combinations(datasets,l):
            comb_set.append(subset)
    return comb_set

# Combines two time series data according to the given combination
def combine_data(comb,path_to_all_train_data):
    out_mat = np.concatenate(tuple([np.loadtxt(path_to_all_train_data[i]) for i in comb]),axis=0)
    # np.random.shuffle(out_mat)
    return out_mat


# Combines the degradation values (takes the average)
def combine_deg(comb,path_to_all_degradation):
    breadth = np.size(np.loadtxt(path_to_all_degradation[comb[0]]))
    temp_mat = np.empty((len(comb),breadth))
    for i,c in enumerate(comb):
        temp_mat[i,:] = np.loadtxt(path_to_all_degradation[c])
    out_mat = 1/len(comb) * np.sum(temp_mat,axis=0)
    return out_mat


# Generates and combines the f-values
def combine_f(comb,path_to_all_training_data,path_to_all_degradation):
    total_genes = np.size(np.loadtxt(path_to_all_degradation[comb[0]]))
    genes_to_train = total_genes    # Genes for which NNs must be generated
    dt = acc_f.dt
    temp_f = np.empty((0,genes_to_train))

    for i in comb:
        x_train = np.loadtxt(path_to_all_training_data[i])
        gamma = np.loadtxt(path_to_all_degradation[i])
        time_steps = np.size(x_train,0)
        f_vec = np.empty((time_steps,genes_to_train))
        for j in range(genes_to_train):
            f_vec[:,j] = acc_f.f(np.transpose(np.array([x_train[:,j]])),gamma[j],dt)
        temp_f = np.concatenate((temp_f,f_vec),axis=0)
    
    return temp_f



if __name__ == '__main__':
    # Select which data you want to train on: monostable or bistable
    stability = 'monostable'
    # file_names = os.listdir(f'ODE Solver\Data for {stability} parameters')
    deg_list, train_data_list = [], []
    # for f in file_names:
    #     if f[0] == 'd':
    #         deg_list.append(f'ODE Solver\Data for {stability} parameters\\{f}')
    #     elif f[0] == 't':
    #         train_data_list.append(f'ODE Solver\Data for {stability} parameters\\{f}')
    #     else:
    #         continue
    if stability == 'monostable':
        with open("train_list_mono_mult.json",'r') as f:
            train_data_list = json.load(f)
        f.close()
        with open("deg_list_mono.json",'r') as f:
            deg_list = json.load(f)
        f.close()
    else:
        with open("train_list_bi_mult.json",'r') as f:
            train_data_list = json.load(f)
        f.close()
        with open("deg_list_bi.json",'r') as f:
            deg_list = json.load(f)
        f.close()
    
    min_size = 7  # Minimum elements in a combination
    num_data = len(train_data_list)
    # combs = generate_combs(min_size,num_data)
    # test_against = list(range(num_data))
    combs = [(0,5,8,12,18)]
    test_against = [20]

    for comb in combs:
        x_train = combine_data(comb,train_data_list)
        f_vec = combine_f(comb,train_data_list,deg_list)
        #deg = combine_deg(comb,deg_list)

        for data in test_against:
            print(f"\nTraining with data from {comb} against {data}")
            check_with = np.loadtxt(train_data_list[data])
            deg = np.loadtxt(deg_list[data])
            train.train_network(x_train,deg,f_vec,check_with.shape[0],x_train.shape[1],x_train.shape[1],
                batch_size=16,epochs=50,plot_gen_data=True,plot_f_vals=False,act_fn='tanh',check_data=check_with)

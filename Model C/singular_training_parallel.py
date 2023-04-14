import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# This file does the same work as "singular_training.py" but in parallel.


import numpy as np
import gc
import training_function as train
import accessory_functions as acc_f
import json
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm


# Different
def topo_from_matrix(w):
    topo_mat = np.empty(w.shape)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            if w[i,j] > 1:
                topo_mat[i,j] = 1
            elif w[i,j] < 1:
                topo_mat[i,j] = -1
            else:
                topo_mat[i,j] = 0
    return topo_mat


def which_topology(topo_mat,permutation_list):
    num_genes = topo_mat.shape[0]
    for topo_num in range(len(permutation_list)):
        if np.sum(topo_mat == permutation_list[topo_num]) == num_genes*num_genes:
            return topo_num
    return -1


def train_one_dataset(input_tuple,max_exponent,batch_size,epochs,plot_gen_data,plot_f_vals,act_fn,num_iters):
    x_train = np.loadtxt(input_tuple[0])
    check_data = np.copy(x_train)
    gamma = np.loadtxt(input_tuple[1])
    param = ""
    for i in input_tuple[0]:
        if i.isnumeric() and int(i) in list(range(10)):
            param += i

    time_steps = np.size(x_train,0)
    # Defaults
    total_genes = np.size(x_train,1)
    genes_to_train = total_genes    # Genes for which NNs must be generated    
    dt = acc_f.dt
    
    f_vec = np.empty((time_steps,genes_to_train))
    for i in range(genes_to_train):
        f_vec[:,i] = acc_f.f(np.transpose(np.array([x_train[:,i]])),gamma[i],dt)


    correct_exp = 2 * np.ones(genes_to_train)  # Starting with an exponent of 2 for all genes
    perm_list = acc_f.permutation_matrices(total_genes)
    ratio_topologies = np.zeros(len(perm_list))
    err = 0
    for iter in range(num_iters):
        interactions, ms_error = train.train_network(x_train,gamma,f_vec,time_steps,total_genes,genes_to_train,correct_exp,batch_size,epochs,plot_gen_data,plot_f_vals,act_fn,check_data)
        err += ms_error
        topology = topo_from_matrix(interactions)
        topo_num = which_topology(topology,perm_list)
        if topo_num == -1:
            continue
        else:
            ratio_topologies[topo_num] += 1/ms_error
    err = 1/num_iters * err

    # Deletions
    del x_train
    del check_data
    del gamma
    del time_steps
    del total_genes
    del correct_exp
    del f_vec
    del perm_list
    del interactions
    
    # Garbage Collection
    gc.collect()

    return param, 1/np.sum(ratio_topologies) * ratio_topologies, err


def train_all_sets(path_to_all_training_data,path_to_all_degradation,max_exponent=6,batch_size=16,epochs=50,plot_gen_data=False,plot_f_vals=False,act_fn='tanh',num_iters=100):
    num_cores = multiprocessing.cpu_count()
    input_list = list(zip(path_to_all_training_data,path_to_all_degradation))
    input_data = tqdm(input_list)

    g_tr = np.loadtxt(path_to_all_degradation[0])
    num_data_sets = len(path_to_all_training_data)
    param_list = []
    for t in path_to_all_training_data:
        param = ""
        for i in t:
            if i.isnumeric() and int(i) in list(range(10)):
                param += str(i)
        param_list.append(int(float(param)))
    
    prob_matrix = np.zeros((num_data_sets,len(acc_f.permutation_matrices(np.size(g_tr)))))
    err_array = np.zeros(num_data_sets)

    processed_data = Parallel(n_jobs=num_cores-1)(delayed(train_one_dataset)(i,max_exponent,batch_size,epochs,plot_gen_data,plot_f_vals,act_fn,num_iters) for i in input_data)

    for idx,p in enumerate(param_list):
        for tup in processed_data:
            if int(tup[0]) == p:
                prob_matrix[idx,:] = tup[1]
                err_array[idx] = tup[2]
                break

    return prob_matrix, err_array



if __name__ == '__main__':
    # Select which data you want to train on: monostable or bistable
    stability = 'monostable'

    deg_list, train_data_list = [], []
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
    
    # print(train_data_list)
    prob_matrix, err_array = train_all_sets(train_data_list,deg_list,plot_gen_data=False,epochs=50,num_iters=15)
    print("Probability Matrix = \n",prob_matrix)
    pth = os.path.join(os.path.dirname(__file__),"ODE Solver",f"Data for {stability} parameters","Prob_Matrix.txt")
    np.savetxt(pth,prob_matrix)
    pth = os.path.join(os.path.dirname(__file__),"ODE Solver",f"Data for {stability} parameters","Error_Array.txt")
    np.savetxt(pth,err_array)

    # ratios, topos = train_all_sets(['training_data.txt'],['degradation'],num_iters=50)
    # for t in range(len(topos)):
    #     print(f"\nProbability of the following topology = {ratios[0,t]}")
    #     print(topos[t])
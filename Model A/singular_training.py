import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# This file trains the system of neural networks on individual time-series data to
# identify the probability of it coming from a particular topology.


import numpy as np
import training_function as train
import accessory_functions as acc_f
import json


def topo_from_matrix(w):
    topo_mat = np.empty(w.shape)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            if w[i,j] > 0:
                topo_mat[i,j] = 1
            elif w[i,j] < 0:
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


def train_one_dataset(x_train,gamma,f_vec,time_steps,total_genes,genes_to_train,batch_size,epochs,plot_gen_data,plot_f_vals,act_fn,check_data,num_iters):
    perm_list = acc_f.permutation_matrices(total_genes)
    ratio_topologies = np.zeros(len(perm_list))
    for iter in range(num_iters):
        print(f"Training {iter+1}-th time")
        interactions, ms_error = train.train_network(x_train,gamma,f_vec,time_steps,total_genes,genes_to_train,batch_size,epochs,plot_gen_data,plot_f_vals,act_fn,check_data)
        topology = topo_from_matrix(interactions)
        print(interactions) #
        topo_num = which_topology(topology,perm_list)
        print(f"Topology found to be number {topo_num}\n")
        if topo_num == -1:
            continue
        else:
            ratio_topologies[topo_num] += 1/ms_error

    return 1/np.sum(ratio_topologies) * ratio_topologies, perm_list


def train_all_sets(path_to_all_training_data,path_to_all_degradation,batch_size=16,epochs=50,plot_gen_data=False,plot_f_vals=False,act_fn='tanh',num_iters=100):
    num_data_sets = len(path_to_all_training_data)

    x_train = np.loadtxt(path_to_all_training_data[0])
    gamma = np.loadtxt(path_to_all_degradation[0])
    time_steps = np.size(x_train,0)

    # Defaults
    total_genes = np.size(x_train,1)
    genes_to_train = total_genes    # Genes for which NNs must be generated    
    dt = acc_f.dt

    f_vec = np.empty((time_steps,genes_to_train))
    for i in range(genes_to_train):
        f_vec[:,i] = acc_f.f(np.transpose(np.array([x_train[:,i]])),gamma[i],dt)
    
    ratios, topos = train_one_dataset(x_train,gamma,f_vec,time_steps,total_genes,genes_to_train,batch_size,epochs,plot_gen_data,plot_f_vals,act_fn,x_train,num_iters)
    num_topos = len(topos)
    prob_matrix = np.empty((num_data_sets,num_topos))
    prob_matrix[0,:] = ratios

    for set in range(1,num_data_sets):
        x_train = np.loadtxt(path_to_all_training_data[set])
        gamma = np.loadtxt(path_to_all_degradation[set])
        time_steps = np.size(x_train,0)
        f_vec = np.empty((time_steps,genes_to_train))
        for i in range(genes_to_train):
            f_vec[:,i] = acc_f.f(np.transpose(np.array([x_train[:,i]])),gamma[i],dt)
        
        ratios, topos = train_one_dataset(x_train,gamma,f_vec,time_steps,total_genes,genes_to_train,batch_size,epochs,plot_gen_data,plot_f_vals,act_fn,x_train,num_iters)
        prob_matrix[set,:] = ratios

    return prob_matrix, topos



if __name__ == '__main__':
    # Select which data you want to train on: monostable or bistable
    stability = 'bistable'
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
    
    #print(train_data_list)
    prob_matrix, topos = train_all_sets(train_data_list,deg_list,plot_gen_data=True,plot_f_vals=True,epochs=50,num_iters=1)
    print(prob_matrix)
    # pth = os.path.join(os.path.dirname(__file__),"ODE Solver",f"Data for {stability} parameters","Prob_Matrix.txt")
    # np.savetxt(pth,prob_matrix)

    # ratios, topos = train_all_sets(['training_data.txt'],['degradation'],num_iters=50)
    # for t in range(len(topos)):
    #     print(f"\nProbability of the following topology = {ratios[0,t]}")
    #     print(topos[t])
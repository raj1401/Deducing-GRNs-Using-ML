import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# This file checks if the NN that trains on one form of bistable data (A-low, B-high)
# can predict the other form (A-high, B-low).
# This code works in parallel

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import training_function as train
import accessory_functions as acc_f
import json


def work_on_one_dataset(input_tuple,batch_size,epochs,plot_gen_data,plot_f_vals,act_fn):
    x_train = np.loadtxt(input_tuple[0])
    x_check = np.loadtxt(input_tuple[1])
    gamma = np.loadtxt(input_tuple[2])
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

    interactions, ms_error = train.train_network(x_train,gamma,f_vec,time_steps,total_genes,genes_to_train,
                                    batch_size,epochs,plot_gen_data,plot_f_vals,act_fn,x_check)
    
    return param, ms_error


def check_bistability(path_to_train_data,path_to_check_data,path_to_deg_data,batch_size=16,epochs=50,plot_gen_data=False,plot_f_vals=False,act_fn="tanh"):
    error_arr = np.zeros(len(path_to_train_data))

    num_cores = multiprocessing.cpu_count()
    input_list = list(zip(path_to_train_data,path_to_check_data,path_to_deg_data))
    input_data = tqdm(input_list)

    param_list = []
    for t in path_to_train_data:
        param = ""
        for i in t:
            if i.isnumeric() and int(i) in list(range(10)):
                param += str(i)
        param_list.append(int(float(param)))

    processed_data = Parallel(n_jobs=num_cores-1)(delayed(work_on_one_dataset)(i,batch_size,epochs,plot_gen_data,plot_f_vals,act_fn) for i in input_data)

    for idx,p in enumerate(param_list):
        for tup in processed_data:
            if int(tup[0]) == p:
                error_arr[idx] = tup[1]
                break
    
    return error_arr



if __name__ == "__main__":

    deg_list, train_data_list, check_data_list = [], [], []
    with open("train_list_bi_mult.json",'r') as f:
        train_data_list = json.load(f)
    f.close()
    with open("train_list_bi_opp_mult.json",'r') as f:
        check_data_list = json.load(f)
    f.close()
    with open("deg_list_bi.json",'r') as f:
        deg_list = json.load(f)
    f.close()

    err_arr = check_bistability(train_data_list,check_data_list,deg_list)
    pth = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for bistable parameters Checker","Error_Array_Bistability_Check.txt")
    np.savetxt(pth,err_arr)

    # Plotting
    tks = []
    pth = os.path.join(os.path.dirname(__file__),"ODE Solver","to_gen_bi.json")
    with open(pth,'r') as f:
        tks = json.load(f)
    f.close()

    plt.plot(err_arr)
    plt.xticks(ticks=range(len(tks)),labels=tks)
    plt.xlabel("Parameter Number")
    plt.ylabel("Mean Squared Error")
    plt.title("Mean Square Error on Training Various Data Sets")
    fig = plt.gcf()
    fig.set_size_inches(19,11)
    plt.savefig(os.path.join("Plots","bistability_check.png"),dpi=100)
    #plt.show()

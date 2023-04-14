import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools


# Defaults
dt = 0.01
# Euler Integration Step
def g_n1(f,g_n,gamma):
    return g_n + dt * (f - gamma*g_n)


def generate_data(initial_val,genes_to_train,gamma,time_steps,f_gen,x_gen,NN_models):
    x_gen[0,:] = initial_val
    for t in range(time_steps):
        #print(f"Generating Data. Progress = {round(t/time_steps*100)} %", end="\r")
        input_val = np.array([x_gen[t,:]])
        for gene in range(genes_to_train):
            NN_model = NN_models[gene]
            
            # To remove the gene-th gene's value
            input_val_gene = np.zeros((1,genes_to_train-1))
            index = 0
            for i in range(genes_to_train):
                if i != gene:
                    input_val_gene[:,index] = input_val[:,i]
                    index += 1

            f_gen[t,gene] = NN_model(tf.convert_to_tensor(input_val_gene, dtype=tf.float64))
            if t != time_steps - 1:
                x_gen[t+1,gene] = g_n1(f_gen[t,gene],x_gen[t,gene],gamma[gene])
    
    # Clearing the model to avoid clutter
    tf.keras.backend.clear_session()
    return x_gen, f_gen


# Function that converts vector y_i to vector f_i using diff. eqn. in Shen et. al. pg.2
def f(y,gma,dt):
    length = np.size(y,0)
    f_vec = np.empty(length)
    for i in range(length-1):
        f_vec[i] = (y[i+1,0] - y[i,0])/dt + gma*y[i,0]
    f_vec[-1] = f_vec[-2]
    return np.transpose(f_vec)


def mean_squared_error(x_train,x_gen):
    m = x_train.shape[0]
    return 1/(2*m) * np.sum(np.square(x_train-x_gen))



def permutation_matrices(num_genes):
    num_edges = num_genes * (num_genes - 1)
    val_vector = -1 * np.ones(num_edges)
    matrices_list = []
    end_loop = False
    while not end_loop:
        matrix = np.zeros((num_genes,num_genes))
        k = 0
        # Constructing the matrix from val_vector
        for i in range(num_genes):
            for j in range(num_genes):
                if i != j:
                    matrix[i,j] = val_vector[k]
                    k += 1
        matrices_list.append(matrix)
        idx = -1
        val_vector[idx] += 1
        while val_vector[idx] > 1:
            val_vector[idx] = -1
            idx -= 1
            if idx < -num_edges:
                end_loop = True
                break
            val_vector[idx] += 1   
    
    return matrices_list


# Generates all possible combinations for exponent values
def generate_exp_combs(max_exp, genes_to_train):
    exp_values = np.arange(1,max_exp+1)
    prod = itertools.product(exp_values, repeat=genes_to_train)
    return list(prod)


def deduce_interactions(x_train,NN_models,genes_to_train,act_fn,plot=False):
    interaction_matrix = np.empty((genes_to_train,genes_to_train))
    for j in range(genes_to_train):
        NN_model = NN_models[j]
        counter = 0
        for i in range(genes_to_train):
            if (i != j):
                theta_1 = NN_model.layers[(1+2*(genes_to_train-1))+counter].get_weights()[0]
                lmda = theta_1[0]
                interaction_matrix[i,j] = lmda
                counter += 1
            else:
                # Hard-Coded no self activity
                interaction_matrix[i,j] = 1
        
        del NN_model
    
    return interaction_matrix

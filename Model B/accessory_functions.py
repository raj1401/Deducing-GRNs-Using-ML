import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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


############# Works for NNs with two hidden layers with nodes=breadth in sub-neural nets #############
# # Gives the derivative at any x_i value
# def der_ij(x_i,theta_1, bias_1, theta_2, bias_2, theta_3, bias_3, derf_val):
#     new_theta_1 = theta_1.flatten()
#     new_theta_3 = theta_3.flatten()
#     for idx,x in enumerate(x_i):
#         a_1 = np.tanh(x*theta_1 + bias_1)
#         a_2 = np.tanh(np.matmul(a_1,theta_2) + bias_2)

#         a_1 = a_1.flatten()
#         a_2 = a_2.flatten()

#         der_a_2 = np.zeros(a_2.shape)

#         for k in range(a_2.shape[0]):
#             net = 0
#             for w in range(new_theta_1.shape[0]):
#                 net += theta_2[w,k] * (1 - a_1[w]*a_1[w])*new_theta_1[w]
            
#             der_a_2[k] = (1 - a_2[k]*a_2[k])*net

#         intermed = new_theta_3 * der_a_2

#         derf_val[idx] = np.sum(intermed)

# def deduce_interactions(x_train,NN_models,genes_to_train,act_fn,plot=False):
#     interaction_matrix = np.empty((genes_to_train,genes_to_train))
#     plt_num = 0
#     for j in range(genes_to_train):
#         NN_model = NN_models[j]
#         # print(len(NN_model.layers))
#         max_val = np.max(x_train[:,j])
#         min_val = 0   # Any gene can have a minimum expression of zero
#         counter = 0
#         for i in range(genes_to_train):
#             if (i != j):
#                 # Parameters of the sub neural network
#                 theta_1 = NN_model.layers[(1+2*(genes_to_train-1))+counter].get_weights()[0]
#                 bias_1 = NN_model.layers[(1+2*(genes_to_train-1))+counter].get_weights()[1]

#                 theta_2 = NN_model.layers[(1+3*(genes_to_train-1))+counter].get_weights()[0]
#                 bias_2 = NN_model.layers[(1+3*(genes_to_train-1))+counter].get_weights()[1]

#                 theta_3 = NN_model.layers[(1+4*(genes_to_train-1))+counter].get_weights()[0]
#                 bias_3 = NN_model.layers[(1+4*(genes_to_train-1))+counter].get_weights()[1]

#                 # # Trying out new method

#                 # interaction_matrix[i,j] = np.matmul(np.matmul(theta_1,theta_2),theta_3)

#                 counter += 1

#                 derf_val = np.empty(x_train.shape[0])
#                 der_ij(x_train[:,i], theta_1, bias_1, theta_2, bias_2, theta_3, bias_3, derf_val)

#                 plt_num += 1
#                 if plot == True:
#                     plt.subplot(genes_to_train,genes_to_train,plt_num)
            
#                 interaction_matrix[i,j] = np.mean(derf_val)
#                 if plot == True:
#                     plt.plot(x_train[:,i],derf_val)

#             else:
#                 # Hard-Coded no self activity
#                 interaction_matrix[i,j] = 0
#                 plt_num += 1
#                 if plot == True:
#                     plt.subplot(genes_to_train,genes_to_train,plt_num)
#                     plt.plot(x_train[:,i],np.zeros(shape=x_train[:,i].shape))
    
#     if plot == True:
#         plt.show()
    
#     return interaction_matrix

###################################################################################################


################# Works for any number of hidden layers in sub-neural nets with nodes=breadth ###########
### Set value of num_HL in deduce_interactions

def der_ij(x_i, weight_list, bias_list, derf_val):
    for idx, x in enumerate(x_i):
        a_vals_list = []

        a_vals_list.append(np.tanh(x*weight_list[0]+bias_list[0]).flatten())

        for lyr in range(1,len(weight_list)-1):
            a_vals_list.append(np.tanh(np.matmul(a_vals_list[lyr-1],weight_list[lyr])+bias_list[lyr]).flatten())
        
        derivative_list = []

        der_a_2 = np.zeros(a_vals_list[1].shape)

        for l in range(a_vals_list[1].shape[0]):
            net = 0
            for k in range(a_vals_list[0].shape[0]):
                net += weight_list[1][k,l] * (1-a_vals_list[0][k]*a_vals_list[0][k]) * weight_list[0][k]
            
            der_a_2[l] = (1 - a_vals_list[1][l]*a_vals_list[1][l]) * net
        
        derivative_list.append(der_a_2)

        for lyr in range(2, len(weight_list)-1):
            der_a = np.zeros(a_vals_list[lyr].shape)

            for l in range(a_vals_list[lyr].shape[0]):
                net = 0
                for k in range(a_vals_list[lyr - 1].shape[0]):
                    net += weight_list[lyr][k,l] * derivative_list[lyr - 2][k]
                
                der_a[l] = (1 - a_vals_list[lyr][l]*a_vals_list[lyr][l]) * net

            derivative_list.append(np.copy(der_a))
    
        derf_val[idx] = np.sum(weight_list[-1] * derivative_list[-1])



def deduce_interactions(x_train,NN_models,genes_to_train,depth,act_fn,plot=False):
    num_HL = depth 
    interaction_matrix = np.empty((genes_to_train,genes_to_train))
    plt_num = 0
    for j in range(genes_to_train):
        NN_model = NN_models[j]
        # print(len(NN_model.layers))
        max_val = np.max(x_train[:,j])
        min_val = 0   # Any gene can have a minimum expression of zero
        counter = 0
        for i in range(genes_to_train):
            if (i != j):
                weight_list, bias_list = [], []

                for lyr in range(2,2+num_HL+1):
                    if (lyr == 2 or lyr == 2+num_HL):
                        weight_list.append(NN_model.layers[(1+lyr*(genes_to_train-1))+counter].get_weights()[0].flatten())
                    else:
                        weight_list.append(NN_model.layers[(1+lyr*(genes_to_train-1))+counter].get_weights()[0])
                    bias_list.append(NN_model.layers[(1+lyr*(genes_to_train-1))+counter].get_weights()[1])
                
                counter += 1

                derf_val = np.empty(x_train.shape[0])
                der_ij(x_train[:,i], weight_list, bias_list, derf_val)

                plt_num += 1
                if plot == True:
                    plt.subplot(genes_to_train,genes_to_train,plt_num)
            
                interaction_matrix[i,j] = np.mean(derf_val)
                if plot == True:
                    plt.plot(x_train[:,i],derf_val)

            else:
                # Hard-Coded no self activity
                interaction_matrix[i,j] = 0
                plt_num += 1
                if plot == True:
                    plt.subplot(genes_to_train,genes_to_train,plt_num)
                    plt.plot(x_train[:,i],np.zeros(shape=x_train[:,i].shape))
    
    if plot == True:
        plt.show()
    
    return interaction_matrix

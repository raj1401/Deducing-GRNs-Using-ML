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
            f_gen[t,gene] = NN_model(tf.convert_to_tensor(input_val, dtype=tf.float64))
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


#####  Three Layered Version - Use when neural nets have total 3 layers (including input layer)  #######

# # Gives the derivative del_f_i/del_x_j
# def der_ij(x_train, theta_1, theta_1_i, bias_1, theta_2, act_fn):
#     derf_val = np.zeros(x_train.shape[0])

#     for idx in range(x_train.shape[0]):
#         x = x_train[idx,:]

#         a_1 = np.tanh(np.matmul(x,theta_1) + bias_1)
#         intermed = np.multiply(theta_1_i, (1 - np.square(a_1)))

#         derf_val[idx] = np.sum(np.multiply(theta_2,intermed))
    
#     return derf_val


# def deduce_interactions(x_train,NN_models,genes_to_train,act_fn,plot=False):
#     interaction_matrix = np.empty((genes_to_train,genes_to_train))
#     plt_num = 0

#     for j in range(genes_to_train):
#         NN_Model = NN_models[j]

#         theta_1 = NN_Model.layers[0].get_weights()[0]
#         bias_1 = NN_Model.layers[0].get_weights()[1]

#         theta_2 = NN_Model.layers[1].get_weights()[0]

#         new_theta_2 = theta_2.flatten()

#         for i in range(genes_to_train):
#             theta_1_i = theta_1[i,:]
#             derf_val = der_ij(x_train, theta_1, theta_1_i, bias_1, new_theta_2, act_fn)
#             interaction_matrix[i,j] = np.mean(derf_val)

#             if plot == True:
#                 plt_num += 1
#                 plt.subplot(genes_to_train,genes_to_train,plt_num)
#                 plt.plot(x_train[:,j], derf_val)
    
#     if plot == True:
#         plt.show()
        
#     return interaction_matrix

########################################################################################################


########################## Works for a general "N" number of layers >= 4 ###############################
def der_ij(x_train, N, theta_1_i, weight_list, bias_list, act_fn):
    derf_val = np.zeros(x_train.shape[0])

    for idx in range(x_train.shape[0]):
        x = x_train[idx,:]

        a_vals_list = []
        a_vals_list.append(np.tanh(np.matmul(x,weight_list[0]) + bias_list[0]).reshape(1,-1).flatten())

        for lyr in range(1,N-2):
            a_vals_list.append(np.tanh(np.matmul(a_vals_list[lyr - 1],weight_list[lyr]) + bias_list[lyr]).flatten())
        
        derivative_list = []

        der_a_2 = np.zeros(a_vals_list[1].shape)

        for l in range(a_vals_list[1].shape[0]):
            net = 0
            for k in range(a_vals_list[0].shape[0]):
                net += weight_list[1][k,l] * (1-a_vals_list[0][k]*a_vals_list[0][k]) * theta_1_i[k]
            
            der_a_2[l] = (1 - a_vals_list[1][l]*a_vals_list[1][l]) * net
        
        derivative_list.append(der_a_2)

        for lyr in range(2, N-2):
            der_a = np.zeros(a_vals_list[lyr].shape)

            for l in range(a_vals_list[lyr].shape[0]):
                net = 0
                for k in range(a_vals_list[lyr - 1].shape[0]):
                    net += weight_list[lyr][k,l] * derivative_list[lyr - 2][k]
                
                der_a[l] = (1 - a_vals_list[lyr][l]*a_vals_list[lyr][l]) * net

            derivative_list.append(np.copy(der_a))
    
        derf_val[idx] = np.sum(weight_list[-1] * derivative_list[-1])

    return derf_val


def deduce_interactions(x_train,NN_models,genes_to_train,act_fn,plot=False):
    interaction_matrix = np.empty((genes_to_train,genes_to_train))
    plt_num = 0

    for j in range(genes_to_train):
        NN_Model = NN_models[j]

        N = len(NN_Model.layers) + 1  # Total number of layers

        weight_list, bias_list = [], []

        for lyr in range(N-1):
            if (lyr == N-2):
                weight_list.append(NN_Model.layers[lyr].get_weights()[0].flatten())
            else:
                weight_list.append(NN_Model.layers[lyr].get_weights()[0])
            bias_list.append(NN_Model.layers[lyr].get_weights()[1])


        for i in range(genes_to_train):
            theta_1_i = weight_list[0][i,:]
            derf_val = der_ij(x_train, N, theta_1_i, weight_list, bias_list, act_fn)
            interaction_matrix[i,j] = np.mean(derf_val)

            if plot == True:
                plt_num += 1
                plt.subplot(genes_to_train,genes_to_train,plt_num)
                plt.plot(x_train[:,j], derf_val)
    
    if plot == True:
        plt.show()
        
    return interaction_matrix

###################################################################################################


################################ Same as Shen et al. 2021 #########################################
# def deduce_interactions(x_train,NN_models,genes_to_train,act_fn,plot=False):
#     fold_change_lmda = 0.95  ## Hyperparameter
#     interaction_matrix = np.empty((genes_to_train,genes_to_train))
#     plt_num = 0

#     for j in range(genes_to_train):
#         NN_Model = NN_models[j]

#         for i in range(genes_to_train):
#             new_x_train = np.copy(x_train)
#             new_x_train[:,i] = fold_change_lmda * new_x_train[:,i]

#             diff_preds = NN_Model(x_train) - NN_Model(new_x_train)
#             interaction_matrix[i,j] = np.mean(diff_preds)

#             if plot == True:
#                 plt_num += 1
#                 plt.subplot(genes_to_train,genes_to_train,plt_num)
#                 plt.plot(x_train[:,j], diff_preds)
    
#     if plot == True:
#         plt.show()
    
#     return interaction_matrix

#################################################################################################


######################## Previous versions for specific number of layers ########################
# # Four Layered Version
# # Gives the derivative del_f_i/del_x_j
# def der_ij(x_train, theta_1, theta_1_i, bias_1, theta_2, bias_2, theta_3, act_fn):
#     derf_val = np.zeros(x_train.shape[0])

#     for idx in range(x_train.shape[0]):
#         x = x_train[idx,:]

#         a_1 = np.tanh(np.matmul(x,theta_1) + bias_1)
#         a_1 = a_1.reshape(1,-1)
#         a_2 = np.tanh(np.matmul(a_1,theta_2) + bias_2)

#         a_1 = a_1.flatten()
#         a_2 = a_2.flatten()

#         der_a_2 = np.zeros(a_2.shape)

#         for l in range(a_2.shape[0]):
#             net = 0
#             for k in range(a_1.shape[0]):
#                 net += theta_2[k,l] * (1-a_1[k]*a_1[k]) * theta_1_i[k]
            
#             der_a_2[l] = (1 - a_2[l]*a_2[l]) * net
        
#         derf_val[idx] = np.sum(theta_3 * der_a_2)
    
#     return derf_val

# # Four Layered Version
# def deduce_interactions(x_train,NN_models,genes_to_train,act_fn,plot=False):
#     interaction_matrix = np.empty((genes_to_train,genes_to_train))
#     plt_num = 0

#     for j in range(genes_to_train):
#         NN_Model = NN_models[j]

#         theta_1 = NN_Model.layers[0].get_weights()[0]
#         bias_1 = NN_Model.layers[0].get_weights()[1]

#         theta_2 = NN_Model.layers[1].get_weights()[0]
#         bias_2 = NN_Model.layers[1].get_weights()[1]

#         theta_3 = NN_Model.layers[2].get_weights()[0]

#         new_theta_3 = theta_3.flatten()

#         for i in range(genes_to_train):
#             theta_1_i = theta_1[i,:]
#             derf_val = der_ij(x_train, theta_1, theta_1_i, bias_1, theta_2, bias_2, new_theta_3, act_fn)
#             interaction_matrix[i,j] = np.mean(derf_val)

#             if plot == True:
#                 plt_num += 1
#                 plt.subplot(genes_to_train,genes_to_train,plt_num)
#                 plt.plot(x_train[:,j], derf_val)

#     if plot == True:
#         plt.show()
        
#     return interaction_matrix




# # Five Layered Version
# # Gives the derivative del_f_i/del_x_j
# def der_ij(x_train, theta_1, theta_1_i, bias_1, theta_2, bias_2, theta_3, bias_3, theta_4, act_fn):
#     derf_val = np.zeros(x_train.shape[0])

#     for idx in range(x_train.shape[0]):
#         x = x_train[idx,:]

#         a_1 = np.tanh(np.matmul(x,theta_1) + bias_1)
#         a_1 = a_1.reshape(1,-1)
#         a_2 = np.tanh(np.matmul(a_1,theta_2) + bias_2)
#         a_3 = np.tanh(np.matmul(a_2,theta_3) + bias_3)

#         a_1 = a_1.flatten()
#         a_2 = a_2.flatten()
#         a_3 = a_3.flatten()

#         der_a_2 = np.zeros(a_2.shape)

#         for l in range(a_2.shape[0]):
#             net = 0
#             for k in range(a_1.shape[0]):
#                 net += theta_2[k,l] * (1-a_1[k]*a_1[k]) * theta_1_i[k]
            
#             der_a_2[l] = (1 - a_2[l]*a_2[l]) * net
        

#         der_a_3 = np.zeros(a_3.shape)

#         for m in range(a_3.shape[0]):
#             net = 0
#             for l in range(a_2.shape[0]):
#                 net += theta_3[l,m] * der_a_2[l]
            
#             der_a_3[m] = (1-a_3[m]*a_3[m]) * net
        
#         derf_val[idx] = np.sum(theta_4 * der_a_3)
    
#     return derf_val

# # Five Layered Version
# def deduce_interactions(x_train,NN_models,genes_to_train,act_fn,plot=False):
#     interaction_matrix = np.empty((genes_to_train,genes_to_train))
#     plt_num = 0

#     for j in range(genes_to_train):
#         NN_Model = NN_models[j]

#         theta_1 = NN_Model.layers[0].get_weights()[0]
#         bias_1 = NN_Model.layers[0].get_weights()[1]

#         theta_2 = NN_Model.layers[1].get_weights()[0]
#         bias_2 = NN_Model.layers[1].get_weights()[1]

#         theta_3 = NN_Model.layers[2].get_weights()[0]
#         bias_3 = NN_Model.layers[2].get_weights()[1]

#         theta_4 = NN_Model.layers[3].get_weights()[0]

#         new_theta_4 = theta_4.flatten()

#         for i in range(genes_to_train):
#             theta_1_i = theta_1[i,:]
#             derf_val = der_ij(x_train, theta_1, theta_1_i, bias_1, theta_2, bias_2, theta_3, bias_3, new_theta_4, act_fn)
#             interaction_matrix[i,j] = np.mean(derf_val)

#             if plot == True:
#                 plt_num += 1
#                 plt.subplot(genes_to_train,genes_to_train,plt_num)
#                 plt.plot(x_train[:,j], derf_val)

#     if plot == True:
#         plt.show()
        
#     return interaction_matrix



# # Six Layered Version
# # Gives the derivative del_f_i/del_x_j
# def der_ij(x_train, theta_1, theta_1_i, bias_1, theta_2, bias_2, theta_3, bias_3, theta_4, bias_4, theta_5, act_fn):
#     derf_val = np.zeros(x_train.shape[0])

#     for idx in range(x_train.shape[0]):
#         x = x_train[idx,:]

#         a_1 = np.tanh(np.matmul(x,theta_1) + bias_1)
#         a_1 = a_1.reshape(1,-1)
#         a_2 = np.tanh(np.matmul(a_1,theta_2) + bias_2)
#         a_3 = np.tanh(np.matmul(a_2,theta_3) + bias_3)
#         a_4 = np.tanh(np.matmul(a_3,theta_4) + bias_4)

#         a_1 = a_1.flatten()
#         a_2 = a_2.flatten()
#         a_3 = a_3.flatten()
#         a_4 = a_4.flatten()

#         der_a_2 = np.zeros(a_2.shape)

#         for l in range(a_2.shape[0]):
#             net = 0
#             for k in range(a_1.shape[0]):
#                 net += theta_2[k,l] * (1-a_1[k]*a_1[k]) * theta_1_i[k]
            
#             der_a_2[l] = (1 - a_2[l]*a_2[l]) * net
        

#         der_a_3 = np.zeros(a_3.shape)

#         for m in range(a_3.shape[0]):
#             net = 0
#             for l in range(a_2.shape[0]):
#                 net += theta_3[l,m] * der_a_2[l]
            
#             der_a_3[m] = (1-a_3[m]*a_3[m]) * net
        

#         der_a_4 = np.zeros(a_4.shape)

#         for p in range(a_4.shape[0]):
#             net = 0
#             for m in range(a_3.shape[0]):
#                 net += theta_4[m,p] * der_a_3[m]
            
#             der_a_4[p] = (1 - a_4[p]*a_4[p]) * net
        
#         derf_val[idx] = np.sum(theta_5 * der_a_4)
    
#     return derf_val

# # Six Layered Version
# def deduce_interactions(x_train,NN_models,genes_to_train,act_fn,plot=False):
#     interaction_matrix = np.empty((genes_to_train,genes_to_train))
#     plt_num = 0

#     for j in range(genes_to_train):
#         NN_Model = NN_models[j]

#         theta_1 = NN_Model.layers[0].get_weights()[0]
#         bias_1 = NN_Model.layers[0].get_weights()[1]

#         theta_2 = NN_Model.layers[1].get_weights()[0]
#         bias_2 = NN_Model.layers[1].get_weights()[1]

#         theta_3 = NN_Model.layers[2].get_weights()[0]
#         bias_3 = NN_Model.layers[2].get_weights()[1]

#         theta_4 = NN_Model.layers[3].get_weights()[0]
#         bias_4 = NN_Model.layers[3].get_weights()[1]

#         theta_5 = NN_Model.layers[4].get_weights()[0]

#         new_theta_5 = theta_5.flatten()

#         for i in range(genes_to_train):
#             theta_1_i = theta_1[i,:]
#             derf_val = der_ij(x_train, theta_1, theta_1_i, bias_1, theta_2, bias_2, theta_3, bias_3, theta_4, bias_4, new_theta_5, act_fn)
#             interaction_matrix[i,j] = np.mean(derf_val)

#             if plot == True:
#                 plt_num += 1
#                 plt.subplot(genes_to_train,genes_to_train,plt_num)
#                 plt.plot(x_train[:,j], derf_val)

#     if plot == True:
#         plt.show()
        
#     return interaction_matrix




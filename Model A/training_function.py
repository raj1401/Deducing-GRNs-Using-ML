import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import accessory_functions as acc_f
import matplotlib.pyplot as plt



# This constraint helps us to restrict any self activity for genes
class SelfActivityRestrictor(tf.keras.constraints.Constraint):

    def __init__(self,num_genes,current_gene,constr_mat):
        self.num_genes = num_genes
        self.current_gene = current_gene
        self.constr_mat = constr_mat

    def __call__(self, w):
        temp_mat = tf.convert_to_tensor(self.constr_mat, dtype=tf.float32)
        subtract_matrix = tf.multiply(w,temp_mat)
        result = w - subtract_matrix
        return result
    
    def get_config(self):
        return {'Constraining Gene': self.current_gene}



def train_network(x_train,gamma,f_vec,time_steps,total_genes,genes_to_train,batch_size,epochs,plot_gen_data,plot_f_vals,act_fn,check_data):
    # Clearing the model to avoid clutter
    tf.keras.backend.clear_session()
    
    #genes_to_train = total_genes    # Genes for which NNs must be generated
    
    # List of all NN models
    NN_models = []
    # Creating the tensors for training
    x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float64)

    for gene in range(genes_to_train):
        constr_mat = np.zeros(shape=(genes_to_train,genes_to_train))
        constr_mat[gene,:] = np.ones(genes_to_train)
        constraint = SelfActivityRestrictor(genes_to_train,gene,constr_mat)

        # 6 Layered Neural Network
        ##################################################################################################
        # Model
        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(x_train_tensor.shape[1],)),
            keras.layers.Dense(genes_to_train, activation=act_fn, kernel_constraint=constraint),
            keras.layers.Dense(genes_to_train, activation=act_fn),
            keras.layers.Dense(genes_to_train, activation=act_fn),
            keras.layers.Dense(genes_to_train, activation=act_fn),
            keras.layers.Dense(1),
        ])

        loss = tf.keras.losses.MeanSquaredError()
        optim = keras.optimizers.Adam(learning_rate=0.01)

        model.compile(loss=loss, optimizer=optim, jit_compile=True)
        #print(model.summary())
        ##################################################################################################

        # Training
        target = tf.convert_to_tensor(f_vec[:,gene], dtype=tf.float64)
        # print(f"Training the network for gene {chr(65+gene)}")
        model.fit(x_train_tensor, target, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0)
        NN_models.append(model)
    
    # print("Generating Data to calculate mean-squared error")

    x_gen, f_gen = acc_f.generate_data(check_data[0,:],genes_to_train,gamma,check_data.shape[0],np.empty(check_data.shape),np.empty(check_data.shape),NN_models)
    error = acc_f.mean_squared_error(check_data,x_gen)
    # print("Mean Squared Error = ",error)

    # Plotting
    if plot_gen_data:
        fig, ax = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
        fig.tight_layout()
        ax[0].set(title='Original Time Series Data',xlabel='Time Steps',ylabel='Gene Expression')
        ax[1].set(title='Data Generated from the System of Neural Networks',xlabel='Time Steps',ylabel='Gene Expression')
        for g in range(genes_to_train):
            ax[0].plot(check_data[:,g],label=f'Gene {g+1}')
            ax[0].legend(loc='upper right')
            ax[1].plot(x_gen[:,g],label=f'Gene {g+1}')
            ax[1].legend(loc='upper right')
        plt.show()

    inter_matrix = acc_f.deduce_interactions(check_data,NN_models,genes_to_train,act_fn=act_fn,plot=plot_f_vals)
    
    return inter_matrix, error

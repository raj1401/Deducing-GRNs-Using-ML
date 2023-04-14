import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import accessory_functions as acc_f
import matplotlib.pyplot as plt



def create_custom_NN(in_shape, genes_to_train, breadth):
    inp_layer = keras.Input(shape = in_shape)
    first_hidden_layer = []

    for g in range(genes_to_train - 1):
        g_th_tensor = tf.reshape(inp_layer[:,g], (-1,1))
        first_hidden_layer.append(keras.layers.Dense(breadth, activation="tanh")(g_th_tensor))
    
    second_hidden_layer = []

    for hl in first_hidden_layer:
        second_hidden_layer.append(keras.layers.Dense(breadth, activation="tanh")(hl))
    
    third_hidden_layer = []

    for hl in second_hidden_layer:
        #third_hidden_layer.append(keras.layers.Dense(1, activation="tanh")(hl))
        third_hidden_layer.append(keras.layers.Dense(1, activation="linear")(hl))
    
    output_layer = keras.layers.Multiply()(third_hidden_layer)

    model = keras.Model(inputs=inp_layer, outputs=output_layer, name="graphNN")
    return model


# Neural Network that identifies individual functions for each edge
class GraphTrainNN(keras.Model):
    def __init__(self, in_shape, genes_to_train, breadth, **kwargs) -> None:
        super(GraphTrainNN, self).__init__(**kwargs)
        self.in_shape = in_shape
        self.genes_to_train = genes_to_train
        self.breadth = breadth
        self.loss_tracker = keras.metrics.MeanSquaredError(name="Loss")
        self.model = None
    
    @property
    def metrics(self):
        return [self.loss_tracker]
    
    def make_model(self):
        self.model = create_custom_NN(self.in_shape,self.genes_to_train,self.breadth)
    
    def my_summary(self):
        return self.model.summary()
    
    def train_step(self, data):
        x, y_true = data
        
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            final_loss = tf.keras.metrics.mean_squared_error(y_true,predictions)

        grads = tape.gradient(final_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(y_true,predictions)        
        return {"loss": self.loss_tracker.result()}




def train_network(x_train,gamma,f_vec,time_steps,total_genes,genes_to_train,batch_size,epochs,plot_gen_data,plot_f_vals,act_fn,check_data):
    # Clearing the model to avoid clutter
    tf.keras.backend.clear_session()
    
    #genes_to_train = total_genes    # Genes for which NNs must be generated
    
    # List of all NN models
    NN_models = []
    # Creating the tensors for training
    #x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float64)

    for gene in range(genes_to_train):
        graph_model = GraphTrainNN(in_shape=(genes_to_train-1,), genes_to_train=genes_to_train, breadth=2)
        graph_model.make_model()
        graph_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01))
        #print(graph_model.my_summary())
        ##################################################################################################
        x_train_tensor = np.zeros((x_train.shape[0],genes_to_train-1))

        index = 0
        for i in range(genes_to_train):
            if i != gene:
                x_train_tensor[:,index] = x_train[:,i]
                index += 1
        
        x_train_tensor = tf.convert_to_tensor(x_train_tensor, dtype=tf.float64)
        ##################################################################################################
        # Training
        target = tf.convert_to_tensor(f_vec[:,gene], dtype=tf.float64)
        # print(f"Training the network for gene {chr(65+gene)}")
        graph_model.fit(x=x_train_tensor, y=target, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0)
        #print(graph_model.model.layers[3].get_weights()[0])
        NN_models.append(graph_model.model)
    
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
            ax[0].plot(check_data[:,g],label=f'Gene {chr(65+g)}')
            ax[0].legend(loc='upper right')
            ax[1].plot(x_gen[:,g],label=f'Gene {chr(65+g)}')
            ax[1].legend(loc='upper right')
        plt.show()

    inter_matrix = acc_f.deduce_interactions(check_data,NN_models,genes_to_train,act_fn=act_fn,plot=plot_f_vals)

    return inter_matrix, error

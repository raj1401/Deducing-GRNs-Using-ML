import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import accessory_functions as acc_f
import matplotlib.pyplot as plt



class GreaterThanZero(keras.constraints.Constraint):
    def __call__(self, w):
        return w * tf.cast(tf.math.greater(w, 0.0), w.dtype)


class HillLayer(keras.layers.Layer):
    def __init__(self, n, constraint) -> None:
        super(HillLayer, self).__init__()
        self.n = n
        self.constraint = constraint

    
    def build(self, input_shape):
        self.w = self.add_weight(shape=(3,), initializer="random_normal", trainable=True, constraint=self.constraint)
        self.b = self.add_weight(shape=(3,), initializer="random_normal", trainable=False)

    
    def call(self, inputs):
        # self.w[0] -> lambda
        # self.w[1] -> basal value
        # self.w[2] -> production rate

        weights = self.w

        #self.n += self.get_change_in_n(inputs)
        denom = tf.math.pow(weights[1], self.n) + tf.math.pow(inputs,self.n)
        numerator = tf.math.pow(weights[1], self.n) + weights[0] * tf.math.pow(inputs,self.n)

        output = weights[2] * numerator / denom

        return output


def create_custom_NN(in_shape, genes_to_train, exponent):
    #non_neg_constraint = tf.keras.constraints.NonNeg()
    non_neg_constraint = GreaterThanZero()
    inp_layer = keras.Input(shape = in_shape)
    first_hidden_layer = []

    for g in range(genes_to_train - 1):
        hill_layer = HillLayer(n=exponent, constraint=non_neg_constraint)
        g_th_tensor = tf.reshape(inp_layer[:,g], (-1,1))
        first_hidden_layer.append(hill_layer(g_th_tensor))
    
    output_layer = keras.layers.Multiply()(first_hidden_layer)

    model = keras.Model(inputs=inp_layer, outputs=output_layer, name="graphNN")
    return model


# Neural Network that identifies individual functions for each edge
class GraphTrainNN(keras.Model):
    def __init__(self, in_shape, genes_to_train, exponent, **kwargs) -> None:
        super(GraphTrainNN, self).__init__(**kwargs)
        self.in_shape = in_shape
        self.genes_to_train = genes_to_train
        self.exponent = exponent
        self.loss_tracker = keras.metrics.MeanSquaredError(name="Loss")
        self.model = None
    
    @property
    def metrics(self):
        return [self.loss_tracker]
    
    def make_model(self):
        self.model = create_custom_NN(self.in_shape,self.genes_to_train,self.exponent)
    
    def my_summary(self):
        return self.model.summary()
    
    def adjust_exponents(self,input_x):
        try:
            for g in range(self.genes_to_train - 1):
                hill_layer = self.model.layers[1 + 2*(self.genes_to_train-1)+g]

                x_vals = np.arange(0,1,0.01,dtype=np.float32)
                val_list = []
                for idx in range(x_vals.shape[0]):
                    intermed = (tf.math.pow(hill_layer.w[1],hill_layer.n) + tf.math.pow(x_vals[idx],hill_layer.n)) * (tf.math.pow(hill_layer.w[1],hill_layer.n) * tf.math.log(hill_layer.w[1]) + hill_layer.w[0] * tf.math.pow(x_vals[idx],hill_layer.n) * tf.math.log(x_vals[idx]))
                    intermed -= (tf.math.pow(hill_layer.w[1],hill_layer.n) + hill_layer.w[0]*tf.math.pow(x_vals[idx],hill_layer.n)) * (tf.math.pow(hill_layer.w[1],hill_layer.n)*tf.math.log(hill_layer.w[1]) + tf.math.pow(x_vals[idx],hill_layer.n)*tf.math.log(x_vals[idx]))
                    intermed = intermed / (tf.math.pow(hill_layer.w[1],hill_layer.n) + tf.math.pow(x_vals[idx],hill_layer.n))**2
                    val_list.append(intermed)
                
                delH_delN = tf.stack(val_list)

                min_val = tf.math.reduce_min(delH_delN)
                max_val = tf.math.reduce_max(delH_delN)
                mid = (min_val+max_val)/2
                min_dist = mid - min_val
                max_dist = max_val - mid

                #print(input_x)
                input_x_arr = tf.cast(input_x[:,g],dtype=tf.float32)
                new_val_list = []
                for idx in range(input_x_arr.shape[0]):
                    intermed = (tf.math.pow(hill_layer.w[1],hill_layer.n) + tf.math.pow(input_x_arr[idx],hill_layer.n)) * (tf.math.pow(hill_layer.w[1],hill_layer.n) * tf.math.log(hill_layer.w[1]) + hill_layer.w[0] * tf.math.pow(input_x_arr[idx],hill_layer.n) * tf.math.log(input_x_arr[idx]))
                    intermed -= (tf.math.pow(hill_layer.w[1],hill_layer.n) + hill_layer.w[0]*tf.math.pow(input_x_arr[idx],hill_layer.n)) * (tf.math.pow(hill_layer.w[1],hill_layer.n)*tf.math.log(hill_layer.w[1]) + tf.math.pow(input_x_arr[idx],hill_layer.n)*tf.math.log(input_x_arr[idx]))
                    intermed = intermed / (tf.math.pow(hill_layer.w[1],hill_layer.n) + tf.math.pow(input_x_arr[idx],hill_layer.n))**2
                    new_val_list.append(intermed)
                
                curr_delH_delN = tf.stack(new_val_list)

                average = tf.math.reduce_mean(curr_delH_delN)

                if (average > mid + 0.5*max_dist):
                    if (hill_layer.n > 1): hill_layer.n -= 1
                elif (average < mid - 0.5*min_dist): hill_layer.n += 1
                else: pass
        except:
            return 0
            
    
    def train_step(self, data):
        x, y_true = data

        # Adjusting the exponents of each layers
        self.adjust_exponents(x)
        
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            final_loss = tf.keras.metrics.mean_squared_error(y_true,predictions)

        grads = tape.gradient(final_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(y_true,predictions)        
        return {"loss": self.loss_tracker.result()}


def train_network(x_train,gamma,f_vec,time_steps,total_genes,genes_to_train,exponent_tup,batch_size,epochs,plot_gen_data,plot_f_vals,act_fn,check_data):
    # Clearing the model to avoid clutter
    tf.keras.backend.clear_session()
    #genes_to_train = total_genes    # Genes for which NNs must be generated
    
    # List of all NN models
    NN_models = []
    # Creating the tensors for training
    #x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float64)

    for gene in range(genes_to_train):
        graph_model = GraphTrainNN(in_shape=(genes_to_train-1,), genes_to_train=genes_to_train, exponent=exponent_tup[gene])
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
        #print(graph_model.model.layers[(1+2*(genes_to_train-1))].get_weights()[0])
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
    
    del NN_models 
    
    return inter_matrix, error

import numpy as np
import os


file_names_mono = os.listdir('Data for monostable parameters')
file_names_bi = os.listdir('Data for bistable parameters')

new_dir_1 = 'Data for monostable parameters Multiple Normalized'
if not os.path.exists(new_dir_1):
    os.makedirs(new_dir_1)
new_dir_2 = 'Data for bistable parameters Multiple Normalized'
if not os.path.exists(new_dir_2):
    os.makedirs(new_dir_2)


# For normalizing opposite bistable data
file_names_bi = os.listdir('Data for bistable parameters Checker')
new_dir_2 = 'Data for bistable parameters Normalized Checker'
if not os.path.exists(new_dir_2):
    os.makedirs(new_dir_2)

for f in file_names_mono:
    if f[0] == 't':
        p = os.path.join(os.path.dirname(__file__),"Data for monostable parameters",f"{f}")
        data = np.loadtxt(p)
        new_dat = np.empty(shape=data.shape)
        new_dat_list = []
        num_genes = data.shape[1]
        param = ""
        for i in f:
                if i.isnumeric():
                    param += i
        param_int = int(param)
        for g in range(num_genes):
            new_dat[:,g] = 1/np.max(data[:,g]) * data[:,g]
        
        if new_dat.shape[0] > 3000:
            new_dat_len = new_dat.shape[0]
            for i in range(0,new_dat_len,int(round(new_dat_len/3000))):
                if i < new_dat_len:
                    new_dat_list.append(list(new_dat[i,:]))
            new_dat = np.array(new_dat_list)

        p = os.path.join(os.path.dirname(__file__),"Data for monostable parameters Multiple Normalized",f"training_data_mult_norm_{param_int}.txt")
        np.savetxt(p,new_dat)


for f in file_names_bi:
    if f[0] == 't':
        p = os.path.join(os.path.dirname(__file__),"Data for bistable parameters",f"{f}")

        # For normalizing opposite bistable data
        p = os.path.join(os.path.dirname(__file__),"Data for bistable parameters Checker",f"{f}")

        data = np.loadtxt(p)
        new_dat = np.empty(shape=data.shape)
        new_dat_list = []
        num_genes = data.shape[1]
        param = ""
        for i in f:
                if i.isnumeric():
                    param += i
        param_int = int(param)
        for g in range(num_genes):
            new_dat[:,g] = 1/np.max(data[:,g]) * data[:,g]
        
        if new_dat.shape[0] > 3000:
            new_dat_len = new_dat.shape[0]
            for i in range(0,new_dat_len,int(round(new_dat_len/3000))):
                if i < new_dat_len:
                    new_dat_list.append(list(new_dat[i,:]))
            new_dat = np.array(new_dat_list)
        
        f_name = "training_data_mult_norm"

        # For normalizing opposite bistable data
        f_name = "training_data_opp"

        p = os.path.join(os.path.dirname(__file__),new_dir_2,f_name+f"_{param_int}.txt")
        np.savetxt(p,new_dat)
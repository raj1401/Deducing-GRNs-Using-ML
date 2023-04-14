import csv
import json
import numpy as np
import os

# The mono array has data for monostable states. First element of each row is the parameter number
# second element is the ratio of steady state level of A and steady state level of B.
# new_mono array has these elements in a sorted order

# The bi array has data for bistable states. First element of each row is the parameter number
# second element is the ratio of states that go to A high versus those that go to B high
# new_bi array has these elements in a sorted order

p = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for monostable parameters","training_data_2.txt")
total_genes = (np.loadtxt(p)).shape[1]

data, params = [], []
mono, bi = [], []

p = os.path.join(os.path.dirname(__file__),"Data","TS_solution.dat")
with open(p,'r') as file:
    dfcsv = csv.reader(file,delimiter='\t')
    for row in dfcsv:
        data.append([eval(elem) for elem in row])
file.close()
data_array = np.array(data)

p = os.path.join(os.path.dirname(__file__),"Data","TS_parameters.dat")
with open(p,'r') as file:
    dfcsv = csv.reader(file,delimiter='\t')
    for row in dfcsv:
        params.append([eval(elem) for elem in row])
file.close()
param_array = np.array(params)


i=0
j=0
while i < (data_array.shape[0]):
    if data_array[i,1] == 1:
        # (A_ss/(g_A/k_A)) / (B_ss/(g_B/k_B))
        mono.append([data_array[i,0],(((2**data_array[i,3])/(param_array[j,2]/param_array[j,2+total_genes]))/((2**data_array[i,4])/(param_array[j,3]/param_array[j,3+total_genes])))])
        # A_ss/B_ss
        # mono.append([data_array[i,0],(((2**data_array[i,3]))/((2**data_array[i,4])))])
    elif data_array[i,1] == 2:
        if data_array[i,3] > data_array[i,4]:
            bi.append([data_array[i,0],(data_array[i,2]/data_array[i+1,2])])
        else:
            bi.append([data_array[i,0],(data_array[i+1,2]/data_array[i,2])])
        i+=1
    else:
        pass
    i+=1
    j+=1


mono = np.array(mono)
bi = np.array(bi)

# Sorting
monoargs = np.argsort(mono[:,1])
biargs = np.argsort(bi[:,1])

new_mono = np.empty(mono.shape)
new_bi = np.empty(bi.shape)

for i in range(mono.shape[0]):
    new_mono[i,:] = mono[monoargs[i],:]

for i in range(bi.shape[0]):
    new_bi[i,:] = bi[biargs[i],:]

to_train_mono, to_train_bi = [], []

i=0
train_mono, train_mono_sing_norm, train_mono_mult_norm, deg_mono = [], [], [], []
while i<new_mono.shape[0]:
    p = int(new_mono[i,0])
    to_train_mono.append(p)
    path1 = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for monostable parameters",f"training_data_{p}.txt")
    path2 = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for monostable parameters Singular Normalized",f"training_data_sing_norm_{p}.txt")
    path3 = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for monostable parameters Multiple Normalized",f"training_data_mult_norm_{p}.txt")
    path4 = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for monostable parameters",f"degradation_{p}.txt")
    train_mono.append(path1)
    train_mono_sing_norm.append(path2)
    train_mono_mult_norm.append(path3)
    deg_mono.append(path4)
    i+=1  # Keep i = 26 to select 30 equally spaced monostable sets

i=0
train_bi, train_bi_sing_norm, train_bi_mult_norm, train_bi_mult_check, deg_bi = [], [], [], [], []
while i<new_bi.shape[0]:
    p = int(new_bi[i,0])
    to_train_bi.append(p)
    path1 = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for bistable parameters",f"training_data_{p}.txt")
    path2 = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for bistable parameters Singular Normalized",f"training_data_sing_norm_{p}.txt")
    path3 = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for bistable parameters Multiple Normalized",f"training_data_mult_norm_{p}.txt")
    path4 = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for bistable parameters",f"degradation_{p}.txt")
    path5 = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for bistable parameters Normalized Checker",f"training_data_opp_{p}.txt")
    train_bi.append(path1)
    train_bi_sing_norm.append(path2)
    train_bi_mult_norm.append(path3)
    deg_bi.append(path4)
    train_bi_mult_check.append(path5)
    i+=1  # Keep i = 7 to select 32 equally spaced bistable sets


# Saving All Files
p = os.path.join(os.path.dirname(__file__),"ODE Solver", "monostable_data.txt")
np.savetxt(p,new_mono)
p = os.path.join(os.path.dirname(__file__),"ODE Solver", "bistable_data.txt")
np.savetxt(p,new_bi)

p = os.path.join(os.path.dirname(__file__),"ODE Solver", "to_gen_mono.json")
with open(p,'w') as f:
    json.dump(to_train_mono,f,indent=2)
f.close()
with open("train_list_mono.json",'w') as f:
    json.dump(train_mono,f,indent=2)
f.close()
with open("train_list_mono_sing.json",'w') as f:
    json.dump(train_mono_sing_norm,f,indent=2)
f.close()
with open("train_list_mono_mult.json",'w') as f:
    json.dump(train_mono_mult_norm,f,indent=2)
f.close()
with open("deg_list_mono.json",'w') as f:
    json.dump(deg_mono,f,indent=2)
f.close()

p = os.path.join(os.path.dirname(__file__),"ODE Solver", "to_gen_bi.json")
with open(p,'w') as f:
    json.dump(to_train_bi,f,indent=2)
f.close()
with open("train_list_bi.json",'w') as f:
    json.dump(train_bi,f,indent=2)
f.close()
with open("train_list_bi_sing.json",'w') as f:
    json.dump(train_bi_sing_norm,f,indent=2)
f.close()
with open("train_list_bi_mult.json",'w') as f:
    json.dump(train_bi_mult_norm,f,indent=2)
f.close()
with open("train_list_bi_opp_mult.json",'w') as f:
    json.dump(train_bi_mult_check,f,indent=2)
f.close()
with open("deg_list_bi.json",'w') as f:
    json.dump(deg_bi,f,indent=2)
f.close()
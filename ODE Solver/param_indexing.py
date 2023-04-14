import numpy as np


####################
# Set these values
file = 'TS.prs'
num_genes = 2
####################

gene_names = [chr(x) for x in range(65,65+num_genes)]
params = []  # First column of file

f = open(file,'r')
for line in f:
    fields = line.split('\t')
    params.append(fields[0])
f.close()
params.pop(0)

# This array indicates where the prod data for i-th gene is present in the parameters file
prod_indices = np.zeros(num_genes,dtype=int)
# This array indicates where the deg data for i-th gene is present in the parameters file
deg_indices = np.zeros(num_genes,dtype=int)
# These are sparse matrices which indicate which index of the parameters file contains data for
# activation or inhibition (respectively) of gene (row) i by gene (column) j
act_param_indices = np.zeros((num_genes,num_genes),dtype=int)
inh_param_indices = np.zeros((num_genes,num_genes),dtype=int)

for i in range(num_genes):
    prod_string = params[i]
    deg_string = params[i+num_genes]
    prod_indices[i] = gene_names.index(prod_string[-1])
    deg_indices[i] = gene_names.index(deg_string[-1]) + num_genes

for j in range(2*num_genes+2,len(params),3):
    param = params[j]
    interaction = param[0:3]
    agent = param[-4]
    target = param[-1]
    agent_index = gene_names.index(agent)
    target_index = gene_names.index(target)
    if interaction == 'Act':
        act_param_indices[target_index,agent_index] = j
    else:
        inh_param_indices[target_index,agent_index] = j


np.savetxt('prod_indices',prod_indices)
np.savetxt('deg_indices',deg_indices)
np.savetxt('act_param_indices',act_param_indices)
np.savetxt('inh_param_indices',inh_param_indices)
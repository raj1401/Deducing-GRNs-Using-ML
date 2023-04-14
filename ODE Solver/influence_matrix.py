import numpy as np


file = 'TS.topo'
num_genes = 2
path_length = 6  # Number of times topology matrix is multiplied to get the influence matrix
gene_names = [chr(x) for x in range(65,65+num_genes)]
topology_matrix = np.zeros((num_genes,num_genes))
content = []  # Contents of the file
f = open(file,'r')
for line in f:
    fields = line.split('\ ')
    content.append(fields[0])
f.close()
content.pop(0)

for line in content:
    i = gene_names.index(line[0])
    j = gene_names.index(line[2])
    val = line[4]
    if val == 1:
        topology_matrix[i,j] = 1
    else:
        topology_matrix[i,j] = -1

print("Topology = \n",np.asarray(topology_matrix,dtype='int32'))

infMat = topology_matrix.copy()
maxMat = topology_matrix.copy()
# Convert the non-zero elements in the matrix to 1 to get Max Matrix
maxMat[maxMat != 0] = 1.0
# Take powers of the matrix to get the influence matrix and update it
for i in range(2,path_length+1):
    a = np.linalg.matrix_power(topology_matrix, i).astype(float)
    b = np.linalg.matrix_power(maxMat, i).astype(float)
    infMat = infMat + np.divide(a, b, out=np.zeros_like(a), where=b!=0)
# Normalise by the path length
infMat = infMat/(path_length)
print("Influence Matrix =\n",infMat)
# influence_matrix = np.linalg.matrix_power(topology_matrix,propagation_dist)
# influence_matrix = np.asarray(influence_matrix,dtype='int32')
# print(influence_matrix)
np.savetxt('influence_matrix',infMat)
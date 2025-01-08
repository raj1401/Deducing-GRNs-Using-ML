# Deducing-GRNs-Using-ML
 "Deducing Gene Regulatory Networks using Machine Learning" - Done by the author, Raj Mehta, as part of his Bachelor's Thesis at the Indian Institute of Science (IISc), Bangalore.

 ![BS Thesis Summary](BS_Thesis_Diagram.jpg)

 ## Abstract
A Gene Regulatory Network (GRN) is a global map of various physical and bio- chemical interactions between genes and gene products. They can be mathematically represented as directed graphs and are usually modeled as a set of coupled Ordinary Differential Equations (ODEs). With the improvement of gene expression profiling tech- niques and the subsequent plethora of gene-expression data generated, the necessity of frameworks that can solve the inverse problem of reconstructing the GRNs reliably from the data is higher now than ever. In this thesis, we propose three machine-learning models with different deep neural network architectures that can deduce GRNs from the gene-expression data assuming the system of genes is modeled as coupled ODEs. We present the benchmarking of these models done on a two-gene system, the toggle switch, and a three-gene system, the toggle triad, which are considered fundamental network motifs. We show that all three models have great performance in predicting the correct GRN (toggle switch/toggle triad) given only the gene-expression data and how they can be improved even more, followed by how such a framework can be used for larger GRNs using edge-deletion and re-training techniques and other possible architectures that can achieve this goal.

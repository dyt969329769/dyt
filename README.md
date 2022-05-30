Deciphering the rules of ribosome binding sites differentiation in context dependence

We used an unsupervised model (One-Class support vector machine, One-Class SVM) to predict the context dependence of the RBSs based on 87-dimensional characteristics of 921 RBS, characterized above. This model is commonly applied to detect outliers in one classification. Because the numbers of HCD (813) and LCD RBSs (108) were unbalanced (8:1), the LCD RBS were considered as outliers. All the input features were generated in silico based on the sequence dataset (table S5).

All the input features were generated in silico based on the sequence dataset. For our model, the A, T, C, or G composition of the RBS, G+C, A+C, A+T, G+T, C+T, and A+G composition of the spacer region, number of two contiguous nucleotides (GT+TG, AC+CA etc.) in the spacer, the 3-mer frequency in the RBS sequence (total of 64 3-mers, such as AAA, AAT, …, GGG),55 the conservation of the SD sequence, the spacer length defined as the length of the region between the SD region and start codon, and average base-pairing probabilities for each position of each sequence, measured by the RNAplfold function of Vienna RNA package version 2.2.4, for the dataset containing 921 mutant RBS variants, were used as the parameters, producing 87-dimensional characteristic vectors. 

Prerequisites
 
Python, NumPy,  SciPy, Matplotlib
A recent NVIDIA GPU
Cuda == 9.0
Python == 2020.3.2

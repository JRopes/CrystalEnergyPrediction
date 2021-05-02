# Crystal Energy Prediction with Invariants

This is the public code repository for my Final Year Project of my Bachelor's degree in Computer Science at the University of Liverpool. The focus of the project is the prediction of crystal energy of crystal structures, based on isometry invariants. The project considered Random Forests, Gaussian processes, and Dense Neural Networks as machine learning methods. The repository consits of the following implementations:

- Prefix "AMD": The three optimized implementations using Average Minimum Distances of crystal structures
- Prefix "DF_T2L_C": The three optimized implementations using the T2L-C variant of the first 8 density functions of crystal structures
- Prefix "DF_T2L_CO": The three optimized implementations using the T2L-CO variant of the first 8 density functions of crystal structures
- "Gaussian_Energy_Filter": the implementation of the prediction landscape reduction method using the Gaussian process with Average Minimum Distances
- "Preprocessing" Folder: Consists of code that is used for the import and preprocessing of the density functions

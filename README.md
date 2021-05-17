# Crystal Energy Prediction with Invariants

This is the public code repository for my Final Year Project of my Bachelor's degree in Computer Science at the University of Liverpool. The focus of the project is the prediction of crystal energy of crystal structures, based on isometry invariants. The project considered Random Forests, Gaussian processes, and Dense Neural Networks as machine learning methods. The repository consits of the following implementations:

- Prefix "AMD_T2L": The three optimized implementations using the T2L variant of the Average Minimum Distances of crystal structures.
- Prefix "AMD_T2L_CON": The three optimized implementations using the T2L-CON variant of the Average Minimum Distances of crystal structures.
- Prefix "DF_T2L_C": The three optimized implementations using the T2L-C variant of the first 8 density functions of crystal structures.
- Prefix "DF_T2L_CO": The three optimized implementations using the T2L-CO variant of the first 8 density functions of crystal structures.
- "Preprocessing" Folder: Consists of code that is used for the import and preprocessing of the density functions.

"AMD_T2L_GaussianProcess_Predictor.ipynb" is the implementation with the highest accuracy, proposed by this project.

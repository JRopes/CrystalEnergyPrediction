import numpy as np 

def DataFiller(data,conditional_value,new_value):
    shape = data.shape
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            for z in range(shape[2]):
                if (data[i,j,z] == conditional_value):
                    data[i,j,z] = new_value
                    
    return data
import numpy as np 
import pandas as pd
import os

def DataFrameFormat(feature_dir_path):
    ##Calculates Dimensions of Dataframe based on Overall Dataset
    max_lines = 0
    max_columns = 0
    counter = 0
    
    for filename in os.listdir(feature_dir_path):
        
        if filename.endswith(".csv"):
            ##Load CSV and count lines
            filepath = os.path.join(feature_dir_path,filename)
            temp_dataframe = pd.read_csv(filepath,header=None)
            data = temp_dataframe.to_numpy()
            
            num_lines= len(temp_dataframe.axes[0])
            num_columns =  len(temp_dataframe.axes[1])
            
            if (num_lines > max_lines): 
                max_lines = num_lines
                max_lines_filename = filename
                x_labels = data[0:,0]
                
            if (num_columns > max_columns): 
                max_columns = num_columns
                
            counter += 1
                
    print("File with greatest Domain: " + max_lines_filename + " || Number of Density Functions: " + str(max_columns - 1))
            
    return (max_lines + 1), (max_columns-1), counter, x_labels

def FindCrystalProperty(job_name, data):
    crystal_properties = data.to_numpy()
    
    search_query = job_name[12:17]
    
    for i in range(len(data.axes[0])):
        
        temp_property_name = str(crystal_properties[i,0])[4:]
        
        if (search_query == temp_property_name):
            return temp_property_name, crystal_properties[i,2]
            break
    
def DataFrameImport(feature_dir_path,label_file_path):
    ##Loads Features and matches in
    (domain, num_functions, num_instances, x_labels) = DataFrameFormat(feature_dir_path)
    
    feature_data = np.zeros((num_instances,num_functions,domain))
    label_data = []
    
    instance_index = 0
    
    crystal_properties = pd.read_csv(label_file_path,header=None)
    
    for filename in os.listdir(feature_dir_path):
        
        label_instance = []
        
        if filename.endswith(".csv"):

            filepath = os.path.join(feature_dir_path,filename)
            df = pd.read_csv(filepath,header=None)
            density_functions = df.to_numpy()
            
            for i in range(1, len(df.axes[1])):
                for j in range(len(df.axes[0])):
                    if(type(density_functions[j,i]) != str):
                        feature_data[instance_index,(i-1),j] = density_functions[j,i]
                    else:
                        feature_data[instance_index,(i-1),j] = 0.0
            
            (crystal_name, crystal_property) = FindCrystalProperty(filename,crystal_properties)
            label_instance.append(crystal_name)
            label_instance.append(crystal_property)
            
            instance_index += 1
            
        label_data.append(label_instance)
        
    return feature_data, label_data, x_labels
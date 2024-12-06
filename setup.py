from pandas import isna
import numpy as np
import pandas as pd
import lhs



def setup_data(assumption_data,correlation_distributions,correlation_parameters,n_samples):


    def inner(row):

        holder = {}

        for item in row.keys():

            if isna(row[item]) !=True:

                holder[item] = row[item]
        
        return holder

    data_holder = {}

    for assumption in assumption_data:

        for data in assumption.index:

            if isna(data) == False:

                row = assumption.loc[data]

                # cleaning the variable names to ensure they are valid python variable names
                data = data.replace(' ','_')
                data = data.replace('.','_')

                if row.__contains__('distribution'):
                    
                    if isna(row['distribution']) != True:

                        data_holder[data], correlation_distributions, correlation_parameters = lhs.lhs_distribution(row,correlation_distributions,correlation_parameters, n_samples)

                    else:

                        data_holder[data] = row
            
                else:

                    data_holder[data] = row
                        
    
    return data_holder



    

    

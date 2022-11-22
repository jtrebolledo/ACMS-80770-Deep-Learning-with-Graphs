from Pooling import *



def Get_LatenteVariableReduce(lat_variable):
    Pooling_Latente_Sum = GraphPooling("Sum")
    Pooling_Latente_Max = GraphPooling("Max")
    Pooling_Latente_Avg = GraphPooling("Mean")

    lat_variable_pooling_sum = Pooling_Latente_Sum(lat_variable)
    lat_variable_pooling_max = Pooling_Latente_Max(lat_variable)
    lat_variable_pooling_avg = Pooling_Latente_Avg(lat_variable)
    
    
    return lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg
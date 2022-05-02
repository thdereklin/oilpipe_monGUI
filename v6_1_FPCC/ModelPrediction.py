# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 15:07:04 2020

@author: Ting-Han
"""
from numpy import zeros, diff, array
from numpy import around, tile, transpose, argwhere
from numpy import empty, append, reshape, squeeze, mean, absolute, column_stack
from numpy import savetxt, sort
from os import mkdir
from os.path import exists
from pandas import read_csv, DataFrame
from scipy.io import savemat
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def ModelPrediction(test_filename, model_filename, start_pred_signal, prederr_signal): 	
    start_pred_signal.emit('處理狀態：模型預測中...')
    
    clear_session()
 
    model_filename = model_filename[0]
    model = load_model(model_filename)
    model.summary()
    
    # Create Folder
    sep_idx = model_filename.rfind('/')
    path = model_filename[:sep_idx+1]
    if not exists(path+'/../output_prediction'):
        mkdir(path+'/../output_prediction')
    
    # Parameters
    par_df = read_csv('./norm_parameter.csv')
    max_val = list(par_df.iloc[0,:])
    max_val = [int(i) for i in max_val]
    
    min_val = list(par_df.iloc[1,:])
    min_val = [int(i) for i in min_val]
    
    layer_names = [layer.name for layer in model.layers]
    layer_shape = [layer.output_shape for layer in model.layers]
    
    inlayer_shape = layer_shape[0]
    inlayer_name = layer_names[0]
    outlayer_name = layer_names[-1]
    
    output_sen = array([int(outlayer_name[-2:])])
    if len(inlayer_name) % 2 == 0:
        input_sen = zeros((int(len(inlayer_name)/2),), dtype = int)
        for i in range(int(len(inlayer_name)/2)):
            input_sen[i] = int(inlayer_name[(i+1)*2-2:(i+1)*2])
        input_sen = sort(input_sen)
    else:
        input_sen = zeros((1,), dtype = int)
        input_sen[0] = int(inlayer_name[0:2])
        
    order = inlayer_shape[0][2]
    
    file_counts = 0
    total_residual = empty([0,1])
    for filename in test_filename:
        df = read_csv(filename)
        
        # Notice
        Data = array(df.iloc[:, 1:len(min_val)+1])
        Data = (Data-tile(min_val,(Data.shape[0],1)))/(tile(max_val,(Data.shape[0],1))-tile(min_val,(Data.shape[0],1)))
        
        Time = array(df.iloc[:,0])
        
        output_sen = squeeze(output_sen)
        target = zeros((len(range(order,df.shape[0],1)), 1))
        time_axis = zeros((len(range(order,df.shape[0],1)) ,1))
        record_vec = zeros((df.shape[0],))
        
        if order != 0:
            input_mat = \
                    zeros((len(range(order,df.shape[0],1)), len(input_sen), order))  

            counts = 0
            for i in range(order, df.shape[0], 1):
                time_window = Time[i-order:i+1]
                if sum(diff(time_window) > 5) >= 1: # Notice
                    continue
                else:
                    record_vec[i] = 1
                    input_mat[counts,:,:] = transpose(Data[i-order:i, squeeze(input_sen)])
                    
                target[counts, :] = Data[i, output_sen]
                time_axis[counts, :] = Time[i]
                counts += 1    
        
            input_mat = input_mat[:counts, :, :]
            target = target[:counts, :]
            time_axis = time_axis[:counts, :]
            
            input_mat = reshape(input_mat,[input_mat.shape[0],input_mat.shape[1],order,1])  
            
        elif order == 0:
            input_mat = \
                    zeros((len(range(order,df.shape[0],1)), len(input_sen), 1))  
            
            counts = 0
            for i in range(order, df.shape[0], 1):
                time_window = Time[i-1:i+1]
                if sum(diff(time_window) > 5) >= 1: # Notice
                    continue
                else:
                    input_mat[counts,:,:] = transpose(Data[i:i+1, squeeze(input_sen)])
                    
                target[counts, :] = Data[i, output_sen]
                time_axis[counts, :] = Time[i]
                counts += 1    
        
            input_mat = input_mat[:counts, :, :]
            target = target[:counts, :]
            time_axis = time_axis[:counts, :]
            
            input_mat = reshape(input_mat,[input_mat.shape[0],input_mat.shape[1],1,1]) 
        
        # Model Prediction
        pred_result = model.predict(input_mat)
        pred_result = pred_result*(max_val[output_sen]-min_val[output_sen])+min_val[output_sen]
        
        target = target*(max_val[output_sen]-min_val[output_sen])+min_val[output_sen]
        residual = pred_result - target
        
        total_residual = append(total_residual, residual, axis = 0)
        
        sep_idx = filename.rfind('/')
        path = filename[:sep_idx+1]
        name = filename[sep_idx:]
        
        dot_idx = name.rfind('.')
        name = name[:dot_idx]

        output_mat = column_stack((time_axis, target, pred_result, residual))
        savemat(path+'/../output_prediction/'+name+'_pred.mat'\
            ,{'pred_result': pred_result, 'target': target, 'residual': residual, 'time_axis': time_axis})
        
        # Check result csv exist
        if exists(path+'/../output_prediction/'+name+'_result.csv'):
            df = read_csv(path+'/../output_prediction/'+name+'_result.csv')
        
        # Prepare OUTPUT csv (return difference value)
        focus_sen = ['新營流量','民雄流量']
        for i in focus_sen:
            sen_vec = zeros((df.shape[0],))
            sen_vec[1:] = diff(df[i])
            df[i+'_diff'] = sen_vec
        
        # Prepare OUTPUT csv (return residual & predict value)
        record_vec = squeeze(argwhere(record_vec.astype(int) == 1))
        if df.columns[output_sen+1] == '新營流量':
            # Process prediction
            temp_data = zeros((df.shape[0],1))            
            temp_data[record_vec] = pred_result
            df['新營流量_pred'] = temp_data
            
            if '民雄流量_pred' in df.columns:
                check_vec = df['民雄流量_pred']
                if sum(check_vec) != 0:
                    pass
            else:
                df['民雄流量_pred'] = zeros((df.shape[0],1))
            
            # Process residual
            temp_data = zeros((df.shape[0],1))
            temp_data[record_vec] = residual
            df['新營流量_resi'] = temp_data      
                        
            if '民雄流量_resi' in df.columns:
                check_vec = df['民雄流量_resi']
                if sum(check_vec) != 0:
                    pass
            else:
                df['民雄流量_resi'] = zeros((df.shape[0],1))
            
        elif df.columns[output_sen+1] == '民雄流量':
            # Process prediction
            temp_data = zeros((df.shape[0],1))
            temp_data[record_vec] = pred_result
            df['民雄流量_pred'] = temp_data
            
            if '新營流量_pred' in df.columns:
                check_vec = df['新營流量_pred']
                if sum(check_vec) != 0:
                    pass
            else:
                df['新營流量_pred'] = zeros((df.shape[0],1))
            
            # Process residual
            temp_data = zeros((df.shape[0],1))
            temp_data[record_vec] = residual
            df['民雄流量_resi'] = temp_data

            if '新營流量_resi' in df.columns:
                check_vec = df['新營流量_resi']
                if sum(check_vec) != 0:
                    pass
            else:
                df['新營流量_resi'] = zeros((df.shape[0],1))
        
        df.to_csv(path+'/../output_prediction/'+name+'_result.csv', encoding='utf-8-sig', index=False)
        
        #%% output .mat file
        output_mat = DataFrame(output_mat)
        output_mat = output_mat.rename(columns={0:'Time',1:'target',2:'prediction',3:'residual'})
        
        output_mat = output_mat.set_index('Time')
        output_mat.to_csv(path+'/../output_prediction/'+name+'_outsen'+str(output_sen+1)+'.csv')
        
        file_counts += 1
        start_pred_signal.emit('處理狀態：已完成 '+str(round((file_counts/len(test_filename))*100))+'%')
    
    savetxt(path+'/../output_prediction/total_residual.csv', total_residual, delimiter=",")
    total_residual = mean(absolute(total_residual))
    total_residual = around(total_residual, decimals = 2)
    
    prederr_signal.emit('預測結果展示 (總體預測誤差：'+str(total_residual)+')')
    
    start_pred_signal.emit('處理狀態：模型預測完成')
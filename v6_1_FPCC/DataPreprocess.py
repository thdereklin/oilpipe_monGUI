# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:24:39 2020

@author: Ting-Han
"""
from csv import writer
from datetime import datetime
from numpy import array, tile, zeros, savez_compressed, diff, transpose, size
from os import mkdir, remove, listdir
from os.path import exists
from pandas import read_csv

def DataPreprocess(inp_sig, sensorList, filenames, start_proc_signal, end_proc_signal):
    start_time = datetime.now()    

    # Parameters
    par_df = read_csv('./norm_parameter.csv')
    max_val = list(par_df.iloc[0,:])
    max_val = [int(i) for i in max_val]
    
    min_val = list(par_df.iloc[1,:])
    min_val = [int(i) for i in min_val]
    
    start_proc_signal.emit('處理狀態：數據轉檔中...')
    order = int(inp_sig[0])
    num_steps = int(inp_sig[1])
    
    output_sen = inp_sig[2]    
    output_sen = array(output_sen)
    
    input_sen = sensorList  
    input_sen = array(input_sen)
    
    file_counts = 0
    total_datasamples = []
    for f in filenames:
        sep_idx = f.rfind('/')
        path = f[:sep_idx+1]
        name = f[sep_idx+1:-4]
        
        # Confirm Correction CSV File Exist
        if exists(path+'/npz_file/'+name+'.npz'):
            end_proc_signal.emit(f+'已被轉檔過，跳過轉檔程序。')
            continue
        
        df = read_csv(path+'/corr_csv/'+name+'_corr.csv')
        Data = array(df.iloc[:, 1:])
        Data[:, :len(min_val)] = (Data[:, :len(min_val)]-tile(min_val,(Data.shape[0],1)))/(tile(max_val,(Data.shape[0],1))-tile(min_val,(Data.shape[0],1)))
            
        Time = array(df.iloc[:,0])
        
        if order != 0:
            input_mat = \
                zeros((len(range(order,df.shape[0],num_steps)), len(input_sen), order))        
            target = zeros((len(range(order,df.shape[0],num_steps)), len(output_sen)))
            
            counts = 0
            for i in range(order, df.shape[0], num_steps):
                time_window = Time[i-order:i+1]
                if sum(diff(time_window) > 5) >= 1:  # Notice
                    continue
                else:
                    input_mat[counts,:,:] = transpose(Data[i-order:i, input_sen])
                target[counts, :] = Data[i, output_sen]
                counts += 1
        elif order == 0:
            input_mat = \
                zeros((len(range(order,df.shape[0],num_steps)), len(input_sen), 1))        
            target = zeros((len(range(order,df.shape[0],num_steps)), len(output_sen)))
            
            counts = 0
            for i in range(order, df.shape[0], num_steps):
                time_window = Time[i-1:i+1]
                if sum(diff(time_window) > 5) >= 1:  # Notice
                    continue
                else:
                    input_mat[counts,:,:] = transpose(Data[i:i+1, input_sen])
                target[counts, :] = Data[i, output_sen]
                counts += 1
        
        if not exists(path+'/npz_file'):
            mkdir(path+'/npz_file')
        
        num_sample = size(input_mat, 0)
        total_datasamples.append([name+'_outsen'+str(output_sen[0]+1)+'.npz', num_sample])
            
        savez_compressed(path+'/npz_file/'+name+'_outsen'+str(output_sen[0]+1)+'.npz', input_mat = input_mat, \
                 target = target, output_sen = output_sen, input_sen = input_sen)
        
        file_counts += 1
        start_proc_signal.emit('處理狀態：已完成 '+str(round((file_counts/len(filenames))*100))+'%')
        end_proc_signal.emit(f+'，轉檔完畢。'+'\n')
    
    if file_counts != 0:
        with open('npzfile_info.csv', 'w', newline='') as csvfile:
            w = writer(csvfile)
            for i in range(len(total_datasamples)):
                w.writerow(total_datasamples[i])
    else:
        pass
        
    existfile = listdir(path+'/npz_file')
    existfile_idx = [0]*len(existfile)
    for f in filenames:
        sep_idx = f.rfind('/')
        name = f[sep_idx+1:-4]
        
        for i in range(len(existfile)):
            if existfile[i] != name+'_outsen'+str(output_sen[0]+1)+'.npz':
                existfile_idx[i] += 1

    if sum(existfile_idx) == max(existfile_idx)*len(existfile_idx):
        pass
    else:
        for i in range(len(existfile)):
            if existfile_idx[i] == max(existfile_idx):          
                remove(path+'/npz_file/'+existfile[i])
    
    start_proc_signal.emit('處理狀態：檔案轉檔完成')
    end_time = datetime.now()
    
    total_runtime = (end_time - start_time).seconds
    end_proc_signal.emit('轉檔執行時間：'+str(total_runtime)+'秒')
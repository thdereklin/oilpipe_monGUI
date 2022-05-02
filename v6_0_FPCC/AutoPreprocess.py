# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:24:39 2020

@author: Ting-Han
"""
import numpy as np
from csv import writer
from datetime import datetime
from numpy import array, tile, zeros, savez_compressed, diff, transpose
from numpy import where, setdiff1d, size
from os import mkdir, listdir
from os.path import exists
from pandas import read_csv, DataFrame

def AutoPreprocess(inp_sig, filenames, start_auto_signal, end_auto_signal):
    start_time = datetime.now()  

    # Parameters
    par_df = read_csv('./norm_parameter.csv')
    max_val = list(par_df.iloc[0,:])
    max_val = [int(i) for i in max_val]
    
    min_val = list(par_df.iloc[1,:])
    min_val = [int(i) for i in min_val]
    
    start_auto_signal.emit('處理狀態：數據轉檔中...')
    order = int(inp_sig[0])
    num_steps = int(inp_sig[1])
    sel_method = int(inp_sig[2])
    
    file_counts = 0
    total_datasamples = []
    for i in range(len(max_val)):
        end_auto_signal.emit('===== 開始產生屬於「'+str(i+1)+'號輸出感測器」的npz檔 ====='+'\n')
        
        output_sen = [i]
        output_sen = array(output_sen)
        
        # Correlation 
        if sel_method == 0:
            input_sen = [i for i in range(len(max_val))] 
            input_sen = array(input_sen)
            input_sen = setdiff1d(input_sen, output_sen)
            
        elif sel_method == 1:
            corr_filenames = ['./../Data/corr_csv/'+_ for _ in listdir('./../Data/corr_csv/') if _.endswith("corr.csv")]
        
            all_df = DataFrame()
            for j in corr_filenames:
                df = read_csv(j)            
                all_df = all_df.append(df)
            
            all_df = all_df.iloc[:,1:]
            corrMatrix = all_df.corr()
            corrMatrix = np.abs(corrMatrix.values)
            
            input_sen = where(corrMatrix[i,:] > 0.5)
            input_sen = input_sen[0]
            
            # if np.sum(array(input_sen)) == i:
            #     pass
            # else:
            #     input_sen = setdiff1d(input_sen, output_sen)
        
            text = ''
            for j in input_sen:
                text += str(j+1)
                text += ', '       
            end_auto_signal.emit('經相關性計算，以第 '+text[:-2]+' 號感測器作為輸入感測器')
        
        # Process each file
        for f in filenames:
            sep_idx = f.rfind('/')
            path = f[:sep_idx+1]
            name = f[sep_idx+1:-4]
            
            # Confirm npz file Exist
            if exists(path+'/npz_file/'+name+'_outsen'+str(output_sen[0]+1)+'.npz'):
                end_auto_signal.emit(f+'已被轉檔過，跳過轉檔程序。')
                continue
            
            df = read_csv(path+'/corr_csv/'+name+'_corr.csv')
            Data = array(df.iloc[:, 1:])
            Data = (Data-tile(min_val,(Data.shape[0],1)))/(tile(max_val,(Data.shape[0],1))-tile(min_val,(Data.shape[0],1)))
                
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
                        
            savez_compressed(path+'/npz_file/'+name+'_outsen'+str(output_sen[0]+1)+'.npz', \
                             input_mat = input_mat, target = target, \
                             output_sen = output_sen, input_sen = input_sen, \
                             num_sample = num_sample)
            
            file_counts += 1
            start_auto_signal.emit('處理狀態：數據轉檔已完成 '+str(round((file_counts/(len(filenames)*len(max_val)))*100))+'%')
            end_auto_signal.emit(f+'，轉檔完畢。'+'\n')
    
    if file_counts != 0:
        with open('npzfile_info.csv', 'w', newline='') as csvfile:
            w = writer(csvfile)
            for i in range(len(total_datasamples)):
                w.writerow(total_datasamples[i])
    else:
        pass
  
    end_time = datetime.now()
    
    start_auto_signal.emit('處理狀態：檔案轉檔完成')
    total_runtime = (end_time - start_time).seconds
    end_auto_signal.emit('轉檔執行時間：'+str(total_runtime)+'秒')
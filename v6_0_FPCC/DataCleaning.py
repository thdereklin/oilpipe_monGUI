# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:17:09 2020

@author: Ting-Han
"""
from csv import writer, reader
from os import mkdir
from os.path import exists
from re import finditer
from pandas import read_csv, DataFrame
from numpy import zeros, diff, argwhere, array, unique, delete, expand_dims, empty
from numpy import intersect1d, interp, around, dtype, squeeze, setdiff1d, concatenate

def DataCleaning(filenames, seninfo_signal, start_clean_signal, end_clean_signal):
    start_clean_signal.emit('處理狀態：數據檢查中...')
    file_counts = 0
    for f in filenames:
        sep_idx = f.rfind('/')
        path = f[:sep_idx+1]
        name = f[sep_idx+1:-4]
        
        # Confirm Correction CSV File Exist
        if exists(path+'/corr_csv/'+name+'_corr.csv'):
            end_clean_signal.emit(f+'已被檢查過，跳過檢查程序。')
            if 'sensor_name' not in locals():
                with open(path+'/corr_csv/sensor_name.csv') as sen_f:
                    csv_reader = reader(sen_f)
                    sensor_name = list(csv_reader)
                    sensor_name = sensor_name[0]
                    seninfo_signal.emit(sensor_name)
            continue
        
        df = read_csv(f)
        sensor_name = list(df.columns)
        sensor_name = sensor_name[1:17]
        seninfo_signal.emit(sensor_name)
        
        time_seq = list(df.iloc[:,0])      
        temp_matrix = df.iloc[:,1:17]
        temp_matrix = temp_matrix.values
        
        null_idx = []
        if temp_matrix.dtype == dtype('float64'):
            data_matrix = temp_matrix
            pass
        else:
            i = 0
            counts = 0
            data_matrix = zeros((temp_matrix.shape[0], temp_matrix.shape[1]))
            while i < len(time_seq):
                if sum(temp_matrix[i,:] == 'BsonNull') > 0:
                    null_idx.append(i)
                    i += 1
                    continue
                else:
                    data_matrix[counts,:] = temp_matrix[i,:].astype(float)
                    i += 1
                    counts += 1
                    
            if counts != 0:
                data_matrix = data_matrix[:counts, :]
            else:
                null_idx = empty([0,], dtype=int)
                
                null_sen = squeeze(argwhere(temp_matrix[0,:] == 'BsonNull'))
                unnull_sen = list(setdiff1d(array(range(len(sensor_name))), null_sen))
                for i in unnull_sen:
                    # Problem BsonNull remove
                    unnull_time = squeeze(argwhere(temp_matrix[:,i] != 'BsonNull'))
                    
                    # Prevent over squeeze, list needs 1-dimension array
                    temp = squeeze(argwhere(temp_matrix[:,i] == 'BsonNull'))
                    if len(temp) == 1:
                        temp = expand_dims(temp, axis = 0)
                    null_idx = concatenate((null_idx, temp))
                    temp_matrix = temp_matrix[unnull_time, :]
                    
                null_idx = list(null_idx)
                
                # Change data type (only unnull)
                for i in unnull_sen:
                    data_matrix[:temp_matrix.shape[0],i] = temp_matrix[:,i].astype(float)
                    
                data_matrix = data_matrix[:temp_matrix.shape[0], :]
                    
                for i in null_sen:
                    end_clean_signal.emit("Warning: 感測器名稱-"+sensor_name[i]+\
                                          "完全斷訊")
        
        date_matrix = zeros((len(time_seq), 6))
        for i in range(0,len(time_seq)):
            timestamp = time_seq[i]             
            if not list(finditer('/', timestamp)):
                date_symbol = [timestamp.start() for timestamp in finditer('-', timestamp)]
            elif not list(finditer('-', timestamp)):
                date_symbol = [timestamp.start() for timestamp in finditer('/', timestamp)]
                
            time_symbol = [timestamp.start() for timestamp in finditer(':', timestamp)]
    
            date_matrix[i,0] = int(timestamp[0:date_symbol[0]])
            date_matrix[i,1] = int(timestamp[date_symbol[0]+1:date_symbol[1]])
            date_matrix[i,2] = int(timestamp[date_symbol[1]+1:date_symbol[1]+3])
    
            date_matrix[i,3] = int(timestamp[time_symbol[0]-2:time_symbol[0]])
            date_matrix[i,4] = int(timestamp[time_symbol[0]+1:time_symbol[1]])
            date_matrix[i,5] = int(timestamp[time_symbol[1]+1:])
    
        orgtime_vec = (date_matrix[:,0]-date_matrix[0,0])*(365*30*24*60*60)+\
            (date_matrix[:,1]-date_matrix[0,1])*(30*24*60*60)+\
            (date_matrix[:,2]-date_matrix[0,2])*(24*60*60)+\
            (date_matrix[:,3]-date_matrix[0,3])*(60*60)+\
            (date_matrix[:,4]-date_matrix[0,4])*60+\
            (date_matrix[:,5]-date_matrix[0,5])
    
        df['Time'] = orgtime_vec
        
        if len(orgtime_vec) != len(null_idx):
            time_vec = delete(orgtime_vec, null_idx, 0)
        else:
            time_vec = orgtime_vec

        # Print Info. (Total Samples)
        end_clean_signal.emit(f+'，共有'+str(len(time_vec))+'筆資料被處理')
    
        time_diff = diff(time_vec)
        missing_period = argwhere(time_diff > 15)
        
        # Print Info. (Missing Period)
        if len(missing_period) == 0:
            end_clean_signal.emit('=> 數據內未有遺失資料之情況')
        else:
            end_clean_signal.emit('=> 時間軸上共有'+str(len(missing_period))+'個時段有遺失資料，'+\
                                  '每個時段皆超過15秒，此部分缺失資料不使用內插法進行補遺。')
        
        # Print Info. (Repeat Samples)
        if len(time_vec)-len(unique(time_vec)) == 0:
            end_clean_signal.emit('=> 數據內未有重複採樣之情況')
        else:
            end_clean_signal.emit('=> 數據中因有'+str(len(time_vec)-len(unique(time_vec)))+'筆資料有重複採樣而被移除。')
    
        max_time = max(time_vec)
        cortime_vec = array(range(int(max_time)+1))
        for i in range(0,len(missing_period)):
            missing_lower = argwhere(cortime_vec > time_vec[int(missing_period[i])])
            missing_upper = argwhere(cortime_vec < time_vec[int(missing_period[i]+1)])
    
            cortime_vec = delete(cortime_vec, intersect1d(missing_lower, missing_upper))
        
        cor_df = DataFrame({'Time': cortime_vec})
        
        for i in range(data_matrix.shape[1]):
            y_corvec = interp(cortime_vec, time_vec, data_matrix[:,i])
            cor_df[df.columns[i+1]] = y_corvec
        
        cor_df = around(cor_df, decimals = 7)
        cor_df = cor_df.set_index('Time')
        
        if not exists(path+'/corr_csv'):
            mkdir(path+'/corr_csv')
        if not exists(path+'/corr_csv/sensor_name.csv'):
            with open(path+'/corr_csv/sensor_name.csv', "w") as sen_f:
                csv_writer = writer(sen_f, delimiter = ',')
                csv_writer.writerow(sensor_name)
            
        cor_df.to_csv(path+'/corr_csv/'+name+'_corr.csv', encoding="utf_8_sig")
        
        file_counts += 1
        start_clean_signal.emit('處理狀態：已完成 '+str(round((file_counts/len(filenames))*100))+'%')        
        end_clean_signal.emit('=> 數據檢查完畢。'+'\n')
        
    start_clean_signal.emit('處理狀態：數據檢查已完成')
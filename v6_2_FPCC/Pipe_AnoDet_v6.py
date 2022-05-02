# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:39:30 2020

@author: Ting-Han
"""
from csv import reader, writer
from datetime import datetime
from pandasModel import pandasModel
from sys import argv, exit
from os import mkdir, listdir, remove, environ
from os.path import exists, isfile
from pandas import read_csv, DataFrame
from numpy import array, size, load, around, amax, int32, ceil
from numpy import empty, append, reshape, squeeze
from scipy.io import savemat, loadmat
from functools import partial
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.backend import clear_session
from tensorflow.python.client import device_lib
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from time import sleep
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QCheckBox, QWidget
from PyQt5.QtWidgets import QMessageBox, QGraphicsScene, QTableWidgetItem, QHBoxLayout
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal, QThread, QTimer, QDir
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QPixmap, QStandardItemModel, QStandardItem, QIcon
from sklearn.model_selection import train_test_split
from seaborn import heatmap

from MainUI import Ui_MainWindow
from PlotUI import Ui_ChildrenForm
from DataCleaning import DataCleaning
from DataPreprocess import DataPreprocess
from AutoPreprocess import AutoPreprocess
from ModelPrediction import ModelPrediction
import numpy as np

stop_train_flag = False

# Restrict GPU memory usage
com_info = device_lib.list_local_devices()
for i in range(len(com_info)):
 	device_type = com_info[i].device_type
     
    # check training with using CPU or GPU. If GPU doesn't exist, use CPU
 	if device_type == 'GPU':
         environ["CUDA_VISIBLE_DEVICES"] = "0"
         config = ConfigProto()
         config.gpu_options.allow_growth = True
         session = InteractiveSession(config=config)
 	else:
         pass

#==================================Training===================================#
def model_training(train_filenames, tpList, train_signal, traintime_signal, on_epoch_end_signal):
    global stop_train_flag
    start_time = datetime.now()
    
    clear_session()
    TrainInfo = InfoCallback(on_epoch_end_signal, stop_train_flag)
    
    # Load npz file 
    fileinfo = []
    with open('./npzfile_info.csv', newline='') as csvfile:
        r = reader(csvfile)
        for row in r:
            fileinfo.append(row)
    
    # Parameter
    num_neuron = int(tpList[0]) # number of neurons
    max_epoch = int(tpList[1])  # number of epochs
    num_filter = int(tpList[2]) # number of filters
    model_name = tpList[4]      # model name
    outsensor_name = tpList[5]  # output sensor name
    batch_size = int(tpList[6]) # batch size
    
    # pick npz files corresponding to chosen output sensor 
    filtered_file = []
    for i in range(len(train_filenames)):
        underline = train_filenames[i].rfind('_')
        dot = train_filenames[i].rfind('.')
        
        sen_name = train_filenames[i][underline+1:dot]
        if sen_name == 'outsen'+outsensor_name:
            filtered_file.append(i)
    
    train_filenames = [train_filenames[i] for i in filtered_file]
    
    file = load(train_filenames[0]) 
    output_sen = file['output_sen']
    input_sen = file['input_sen']
    temp_mat = file['input_mat']
    
    # Find Parameters
    num_sensor = size(temp_mat, 1)
    num_order = size(temp_mat, 2)
        
    # Create Folder
    sep_idx = train_filenames[0].rfind('/')
    path = train_filenames[0][:sep_idx+1]
    if not exists(path+'/../model'):
        mkdir(path+'/../model')
    
    # train & validation split
    if len(train_filenames) == 1:
        input_mat = temp_mat
        target = file['target']
        
        num_event = size(input_mat, 0)
        input_mat = reshape(input_mat,[num_event,num_sensor,num_order,1])
        train_input, val_input, train_target, val_target = train_test_split(input_mat, \
                                                                            target, test_size = 0.25)    
    else:
        train_name, val_name = train_test_split(train_filenames, test_size = 0.25)
    
    # Decide Output Layer Name
    inlayer_name = ''
    for i in input_sen:
        inlayer_name += str(int(i)).zfill(2)
    
    outlayer_name = str(int(output_sen)).zfill(2)
    
    if inlayer_name == outlayer_name:
        inlayer_name = inlayer_name+'_in'
      
    # Buildup Model
    input_ = Input(shape = (num_sensor,num_order,1), name = inlayer_name)
    f = Conv2D(filters = num_filter,
               input_shape=(num_sensor,num_order,1),
               kernel_size=(3,3),
               padding='same',
               activation='relu')(input_) 
    f = Conv2D(filters = num_filter,
               input_shape=(num_sensor,num_order,1),
               kernel_size=(3,3),
               padding='same',
               activation='relu')(f)  
    f = GlobalAveragePooling2D()(f)
    f = Dense(num_neuron,activation='relu')(f)
    pred = Dense(1,activation='linear', name = outlayer_name)(f)
    
    model = Model(inputs = input_, outputs = pred)
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    model.summary()

    # Checkpoint
    filepath = path+'/../model/'+model_name+'.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='min')
    
    if len(train_filenames) == 1:
        callbacks_list = [checkpoint, TrainInfo]
        train_history = model.fit(train_input, train_target, epochs=max_epoch, callbacks=callbacks_list, 
                              batch_size=128, validation_data=(val_input,val_target))
    else:
        # Count steps_per_epoch
        num_trainsteps = 0
        for i in range(len(train_name)):
            slash = train_name[i].rfind('/')
            name = train_name[i][slash+1:]
            
            for j in range(len(fileinfo)):
                if fileinfo[j][0] == name:
                    num_trainsteps += ceil(int(fileinfo[j][1])/batch_size)
        
        num_valsteps = 0
        for i in range(len(val_name)):
            slash = val_name[i].rfind('/')
            name = val_name[i][slash+1:]
            
            for j in range(len(fileinfo)):
                if fileinfo[j][0] == name:
                    num_valsteps += ceil(int(fileinfo[j][1])/batch_size)
        
        callbacks_list = [checkpoint, TrainInfo]
        
        # fit model
        train_history = model.fit_generator(generator = get_batch(train_name, batch_size, num_order, input_sen, output_sen, inlayer_name, outlayer_name),  
                                    steps_per_epoch = num_trainsteps,
                                    epochs = max_epoch, 
                                    verbose = 1,
                                    callbacks = callbacks_list, 
                                    validation_data = get_batch(val_name, batch_size, num_order, input_sen, output_sen, inlayer_name, outlayer_name),
                                    validation_steps = num_valsteps)
        
    # Save Training History
    train_loss = array(train_history.history['loss'])
    val_loss = array(train_history.history['val_loss'])
    savemat('train_history.mat',{'train_loss': train_loss, 'val_loss': val_loss})
    
    final_train_loss = around(train_loss[-1], decimals = 4)
    final_val_loss = around(val_loss[-1], decimals = 4)
    train_signal.emit('訓練成效圖 (訓練誤差：'+str(final_train_loss)+'、驗證誤差：'+str(final_val_loss)+')')
    
    end_time = datetime.now()
    total_runtime = (end_time - start_time).seconds
    traintime_signal.emit('訓練時間：'+str(total_runtime)+'秒')
    
def Animation(on_signal, off_signal):
    i = 0
    while i >= 0:
        on_signal.emit(True)
        sleep(1)
        off_signal.emit(False)
        sleep(1)
        
class InfoCallback(Callback):
    def __init__(self, signal, stop_train_flag):
        Callback.__init__(self)
        self.train_err = []
        self.val_err = []
        self.signal = signal
        self.stop_train_flag = stop_train_flag
        
    def on_epoch_end(self, epoch, logs = {}):
        self.train_err.append(logs.get('acc'))
        self.val_err.append(logs.get('val_acc'))
        self.signal.emit(epoch, [self.train_err, self.val_err])
        
        if stop_train_flag == True:
            self.model.stop_training = True
            clear_session()
            
#================================Auto Training================================#
def auto_training(tpList, end_auto_signal):
    for i in range(len(com_info)):
        device_type = com_info[i].device_type
        if device_type == 'GPU':
            environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            pass
        
    global stop_train_flag
    
    clear_session()
    
    # Load npz file information
    fileinfo = []
    with open('./npzfile_info.csv', newline='') as csvfile:
        r = reader(csvfile)
        for row in r:
            fileinfo.append(row)
            
    # Load sensor information
    seninfo = []
    with open('./norm_parameter.csv', newline='') as csvfile:
        r = reader(csvfile)
        for row in r:
            seninfo.append(row)
    
    # Parameter
    num_neuron = int(tpList[0])
    max_epoch = int(tpList[1])
    num_filter = int(tpList[2])
    batch_size = int(tpList[4])
    
    path = './../Data/npz_file/'
    
    for i in range(len(seninfo[0])):
        end_auto_signal.emit('===== 開始訓練「'+str(i+1)+'號輸出感測器」的模型 =====')
        
        model_name = 'model_'+str(i+1).zfill(2)
        outsensor_name = str(i+1) 
        
        train_filenames = [path+_ for _ in listdir(path) if _.endswith(".npz")]
    
        filtered_file = []
        for i in range(len(train_filenames)):
            underline = train_filenames[i].rfind('_')
            dot = train_filenames[i].rfind('.')
            
            sen_name = train_filenames[i][underline+1:dot]
            if sen_name == 'outsen'+outsensor_name:
                filtered_file.append(i)
        
        train_filenames = [train_filenames[i] for i in filtered_file]
        
        file = load(train_filenames[0]) 
        output_sen = file['output_sen']
        input_sen = file['input_sen']
        temp_mat = file['input_mat']
        
        # Find Parameters
        num_sensor = size(temp_mat, 1)
        num_order = size(temp_mat, 2)
            
        # Create Folder
        sep_idx = train_filenames[0].rfind('/')
        path = train_filenames[0][:sep_idx+1]
        if not exists(path+'/../model'):
            mkdir(path+'/../model')
    
        # train & validation split
        if len(train_filenames) == 1:
            input_mat = temp_mat
            target = file['target']
            
            num_event = size(input_mat, 0)
            input_mat = reshape(input_mat,[num_event,num_sensor,num_order,1])
            train_input, val_input, train_target, val_target = train_test_split(input_mat, \
                                                                                target, test_size = 0.25)
        else:
            train_name, val_name = train_test_split(train_filenames, test_size = 0.25)
          
        # Decide Output Layer Name
        inlayer_name = ''
        for i in input_sen:
            inlayer_name += str(int(i)).zfill(2)
        
        outlayer_name = str(int(output_sen)).zfill(2)
        
        if inlayer_name == outlayer_name:
            inlayer_name = inlayer_name+'_in'
          
        # Buildup Model
        input_ = Input(shape = (num_sensor,num_order,1), name = inlayer_name)
        f = Conv2D(filters = num_filter,
                   input_shape=(num_sensor,num_order,1),
                   kernel_size=(3,3),
                   padding='same',
                   activation='relu')(input_) 
        f = Conv2D(filters = num_filter,
                   input_shape=(num_sensor,num_order,1),
                   kernel_size=(3,3),
                   padding='same',
                   activation='relu')(f)  
        f = GlobalAveragePooling2D()(f)
        f = Dense(num_neuron,activation='relu')(f)
        pred = Dense(1,activation='linear', name = outlayer_name)(f)
        
        model = Model(inputs = input_, outputs = pred)
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        model.summary()
        
        # Checkpoint
        filepath = path+'/../model/'+model_name+'.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='min')
        
        callbacks_list = [checkpoint]
        
        if len(train_filenames) == 1:
            train_history = model.fit(train_input, train_target, epochs=max_epoch, callbacks=callbacks_list, 
                                  batch_size=128, validation_data=(val_input,val_target))
        else:
            # Count steps_per_epoch
            num_trainsteps = 0
            for i in range(len(train_name)):
                slash = train_name[i].rfind('/')
                name = train_name[i][slash+1:]
                
                for j in range(len(fileinfo)):
                    if fileinfo[j][0] == name:
                        num_trainsteps += ceil(int(fileinfo[j][1])/batch_size)
            
            num_valsteps = 0
            for i in range(len(val_name)):
                slash = val_name[i].rfind('/')
                name = val_name[i][slash+1:]
                
                for j in range(len(fileinfo)):
                    if fileinfo[j][0] == name:
                        num_valsteps += ceil(int(fileinfo[j][1])/batch_size)  
            
            # Fit model
            train_history = model.fit_generator(generator = get_batch(train_name, batch_size, num_order, input_sen, output_sen, inlayer_name, outlayer_name),  
                                        steps_per_epoch = num_trainsteps-1,
                                        epochs = max_epoch, 
                                        verbose = 1,
                                        callbacks = callbacks_list, 
                                        validation_data = get_batch(val_name, batch_size, num_order, input_sen, output_sen, inlayer_name, outlayer_name),
                                        validation_steps = num_valsteps-1)
            
        # Save Training History
        train_loss = array(train_history.history['loss'])
        val_loss = array(train_history.history['val_loss'])
        
        min_train_loss = round(min(train_loss), 5)
        min_val_loss = round(min(val_loss), 5)
        end_auto_signal.emit('最小訓練誤差： '+str(min_train_loss)+'、最小驗證誤差： '+str(min_val_loss)+'\n')
        
        savemat('train_history.mat',{'train_loss': train_loss, 'val_loss': val_loss})

def get_batch(filename, batch_size, order, input_sen, output_sen, inlayer_name, outlayer_name):
    np.random.shuffle(filename)
    while 1:
        for i in range(0, len(filename)):
            file = load(filename[i])
            
            x = file['input_mat']
            num_event = size(x, 0)
            x = reshape(x, [num_event,len(input_sen),order,1])
            
            y = file['target']
            
            k = 0
            for j in range(int(ceil(num_event/batch_size))):
                if k+batch_size > num_event:
                    x_batch = x[k:,:,:,:]                
                    y_batch = y[k:]
                else:
                    x_batch = x[k:k+batch_size,:,:,:]                
                    y_batch = y[k:k+batch_size]
                
                k += batch_size
                
                yield({inlayer_name: x_batch}, {outlayer_name: y_batch})

#===================================Workers===================================#
class TrainWorker(QObject):
    started = pyqtSignal()
    finished = pyqtSignal()
    train_signal = pyqtSignal(str)
    traintime_signal = pyqtSignal(str)
    epoch_end_signal = pyqtSignal(int, list)
    
    @pyqtSlot(str)
    def start_task(self, train_filenames, tpList):
        self.started.emit()
        model_training(train_filenames, tpList, self.train_signal, self.traintime_signal, self.epoch_end_signal)
        self.finished.emit()

class PreprocessWorker(QObject):
    started = pyqtSignal()
    finished = pyqtSignal()
    refreshed = pyqtSignal()
    start_proc_signal = pyqtSignal(str)
    end_proc_signal = pyqtSignal(str)
    
    @pyqtSlot(str)
    def start_task(self, inp_sig, sensorList, filenames):
        self.refreshed.emit()
        self.started.emit()
        DataPreprocess(inp_sig, sensorList, filenames, self.start_proc_signal, self.end_proc_signal)
        self.finished.emit()

class AutoWorker(QObject):
    started = pyqtSignal()
    finished = pyqtSignal()
    refreshed = pyqtSignal()
    start_auto_signal = pyqtSignal(str)
    end_auto_signal = pyqtSignal(str)
    
    @pyqtSlot(str)
    def start_task(self, inp_sig, tpList, filenames):
        self.refreshed.emit()
        self.started.emit()
        AutoPreprocess(inp_sig, filenames, self.start_auto_signal, self.end_auto_signal)
        auto_training(tpList, self.end_auto_signal)
        self.finished.emit()
        
class CleanWorker(QObject):
    started = pyqtSignal()
    finished = pyqtSignal()
    refreshed = pyqtSignal()
    start_clean_signal = pyqtSignal(str)
    end_clean_signal = pyqtSignal(str)
    seninfo_signal = pyqtSignal(list)
    
    @pyqtSlot(str)
    def start_task(self, filenames):
        self.refreshed.emit()
        self.started.emit()
        DataCleaning(filenames, self.seninfo_signal, \
                     self.start_clean_signal, self.end_clean_signal)
        self.finished.emit()
        
class PredictWorker(QObject):
    started = pyqtSignal()
    finished = pyqtSignal()
    refreshed = pyqtSignal()
    start_pred_signal = pyqtSignal(str)
    prederr_signal = pyqtSignal(str)
    
    @pyqtSlot(str)
    def start_task(self, test_filename, model_filename):
        # Check Existing Output Mat File
        path = './../Data/output_prediction/'
        if not exists(path):
            pass
        else:
            matfile = [_ for _ in listdir(path) if _.endswith(".mat")]
            if len(matfile) == 0:
                pass
            else:
                for i in range(len(matfile)):
                    remove(path+matfile[i])
        
        self.refreshed.emit()
        self.started.emit()
        ModelPrediction(test_filename, model_filename, self.start_pred_signal, self.prederr_signal)
        self.finished.emit()
        
class AnimationWorker(QObject):
    on_signal = pyqtSignal(bool)
    off_signal = pyqtSignal(bool)
    
    @pyqtSlot(str)
    def start_task(self):
        Animation(self.on_signal, self.off_signal)

#===================================Figures===================================#      
class Figure_Canvas(FigureCanvas):   
    def __init__(self, parent=None, width=4.5, height=2.2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=100)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)
        
    def plotData(self, time, data):
        self.axes.plot(time, data)
        
        
class Epoch_Canvas(FigureCanvas):   
    def __init__(self, parent=None, width=7.25, height=3.5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=80)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)
        
    def plotData(self, time, data):
        rcParams['axes.titleweight'] = 'bold'
        rcParams['font.family'] = 'Arial'
        legend_properties = {'weight':'bold'}
        
        self.axes.plot(time, data[0,:], label = 'Train')
        self.axes.plot(time, data[1,:], label = 'Valid')
        x_locs = self.axes.get_xticks()
        y_locs = self.axes.get_yticks()
        x_locs = around(x_locs, decimals=1)
        y_locs = around(y_locs, decimals=3)
            
        self.axes.set_xticklabels(x_locs, {'fontweight': rcParams['axes.titleweight']})
        self.axes.set_yticklabels(y_locs, {'fontweight': rcParams['axes.titleweight']})
        self.axes.set_xlabel('Epoch', {'fontweight': rcParams['axes.titleweight']})
        self.axes.set_ylabel('Loss', {'fontweight': rcParams['axes.titleweight']})
        self.axes.legend(loc = "upper right", prop = legend_properties)
         
class CorrFigure_Canvas(FigureCanvas):   
    def __init__(self, parent=None, width=9.5, height=4.4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=100)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)
        
    def plotData(self, data):
        rcParams['axes.titleweight'] = 'bold'
        rcParams['font.family'] = 'Arial'
        
        corrMatrix = data.corr()
        
        x_axis_labels = [i for i in range(1, len(data.columns)+1)]
        y_axis_labels = x_axis_labels
        
        map_fig = heatmap(corrMatrix, vmin = -1, vmax = 1, annot=True, \
                          annot_kws={"fontsize":8}, ax = self.axes, cmap = 'rainbow', cbar=False)
        map_fig.set_xticklabels(x_axis_labels,{'fontweight': rcParams['axes.titleweight']}, rotation=0)
        map_fig.set_yticklabels(y_axis_labels,{'fontweight': rcParams['axes.titleweight']})

class Predict_Canvas(FigureCanvas):   
    def __init__(self, parent=None, width=9, height=3, dpi=80):
        fig = Figure(figsize=(width, height), dpi=80)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)
        
    def plotData(self, time, data, legend_label):
        rcParams['axes.titleweight'] = 'bold'
        legend_properties = {'weight':'bold'}
        
        if amax(time) > 86400:
            time = time/86400
            self.axes.set_xlabel('Time (Day)', {'fontweight': rcParams['axes.titleweight']})
        else:
            time = time/3600
            self.axes.set_xlabel('Time (Hour)', {'fontweight': rcParams['axes.titleweight']})
        
        self.axes.plot(time, data, label = legend_label)
        x_locs = self.axes.get_xticks()
        y_locs = self.axes.get_yticks()
        x_locs = int32(x_locs)
        y_locs = around(y_locs, decimals=1)
        
        self.axes.set_xticklabels(x_locs, {'fontweight': rcParams['axes.titleweight']})
        self.axes.set_yticklabels(y_locs, {'fontweight': rcParams['axes.titleweight']})
        self.axes.set_ylabel('Value', {'fontweight': rcParams['axes.titleweight']})
        self.axes.legend(loc = "upper right", prop = legend_properties)
    
    def setting(self):
        rcParams['axes.titleweight'] = 'bold'
        x_locs = self.axes.get_xticks()
        y_locs = self.axes.get_yticks()
        x_locs = around(x_locs, decimals=1)
        y_locs = around(y_locs, decimals=1)
        self.axes.set_xticklabels(x_locs, {'fontweight': rcParams['axes.titleweight']})
        self.axes.set_yticklabels(y_locs, {'fontweight': rcParams['axes.titleweight']})
        
class PredictSum_Canvas(FigureCanvas):   
    def __init__(self, parent=None, width=9, height=3, dpi=80):
        fig = Figure(figsize=(width, height), dpi=80)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)
        
    def plotData(self, data, legend_label):
        rcParams['axes.titleweight'] = 'bold'
        legend_properties = {'weight':'bold'}
        
        self.axes.plot(data, label = legend_label)
        y_locs = self.axes.get_yticks()
        y_locs = around(y_locs, decimals=1)
        
        self.axes.set_xticks([])
        self.axes.set_yticklabels(y_locs, {'fontweight': rcParams['axes.titleweight']})
        self.axes.set_xlabel('Time', {'fontweight': rcParams['axes.titleweight']})
        self.axes.set_ylabel('Value', {'fontweight': rcParams['axes.titleweight']})
        self.axes.legend(loc = "upper right", prop = legend_properties)
        
    def setting(self):
        rcParams['axes.titleweight'] = 'bold'
        x_locs = self.axes.get_xticks()
        y_locs = self.axes.get_yticks()
        x_locs = around(x_locs, decimals=1)
        y_locs = around(y_locs, decimals=1)
        self.axes.set_xticklabels(x_locs, {'fontweight': rcParams['axes.titleweight']})
        self.axes.set_yticklabels(y_locs, {'fontweight': rcParams['axes.titleweight']})

#===================================Main UI===================================#
class MyMainWindow(QMainWindow, Ui_MainWindow):
    browseSignal = pyqtSignal(list)
    testbrowseSignal = pyqtSignal(list)
    modelbrowseSignal = pyqtSignal(list)
    viewSignal = pyqtSignal(list)
    processSignal = pyqtSignal(list)
    trainparSignal = pyqtSignal(list)
    sennameSignal = pyqtSignal(list)
    
    train_flagSignal = pyqtSignal(bool)
    
    def __init__(self, parent=None):    
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.initWorker()
        self.initUI()
        self.child = ChildrenForm(parent=self)
        
        # Set Window Title
        self.setWindowTitle("Pipe Monitoring")
        
        # Set Window Icon
        self.setWindowIcon(QIcon('./valve.png'))
        
        # Hide Tab
        self.tabWidget.removeTab(5)
    
    def initWorker(self):
        # Thread (Model Training)
        self.TrainWorker = TrainWorker()
        self.thread_trn = QThread(self)
        self.thread_trn.start()
        
        self.TrainWorker.moveToThread(self.thread_trn)
        self.progressBar.setValue(0)
        
        self.trainButton.clicked.connect(self.emitTrainparSignal)
        
        self.TrainWorker.train_signal.connect(self.showTrainError)
        self.TrainWorker.traintime_signal.connect(self.showTrainTime)
        self.TrainWorker.epoch_end_signal.connect(self.showProgress)
        self.TrainWorker.started.connect(self.progressBar.show)
        self.TrainWorker.finished.connect(self.epochfigureDisplay)
        
        self.TrainWorker.started.connect(partial(self.trainButton.setEnabled, False))
        self.TrainWorker.finished.connect(partial(self.trainButton.setEnabled, True))
        
        # Thread (Data Preprocessing)
        self.PreprocessWorker = PreprocessWorker()
        thread_preproc = QThread(self)
        thread_preproc.start()
        
        self.PreprocessWorker.moveToThread(thread_preproc)
        self.executeButton.clicked.connect(self.get_targetSensor)
        self.executeButton.clicked.connect(self.emitProcessSignal)
        # self.executeButton.clicked.connect(self.save_norminfo)
        
        self.PreprocessWorker.start_proc_signal.connect(self.showProcingLabel)
        self.PreprocessWorker.end_proc_signal.connect(self.showProcingTextEdit)
        self.PreprocessWorker.refreshed.connect(self.cleanProcingTextEdit)
        self.PreprocessWorker.started.connect(partial(self.executeButton.setEnabled, False))
        self.PreprocessWorker.finished.connect(partial(self.executeButton.setEnabled, True))
        self.PreprocessWorker.started.connect(partial(self.browseButton.setEnabled, False))
        self.PreprocessWorker.finished.connect(partial(self.browseButton.setEnabled, True))
        
        # Thread (Auto Preprocessing)
        self.AutoWorker = AutoWorker()
        thread_auto = QThread(self)
        thread_auto.start()
        
        self.autoButton.clicked.connect(self.emitAutoSignal)
        
        self.AutoWorker.moveToThread(thread_auto)
        self.AutoWorker.start_auto_signal.connect(self.showAutoLabel)
        self.AutoWorker.end_auto_signal.connect(self.showAutoTextEdit)
        self.AutoWorker.refreshed.connect(self.cleanAutoTextEdit)
        
        # Thread (Data Cleaning)
        self.CleanWorker = CleanWorker()
        thread_dataclean = QThread(self)
        thread_dataclean.start()
        
        self.CleanWorker.moveToThread(thread_dataclean)
        self.checkButton.clicked.connect(self.emitCleanSignal)
        self.CleanWorker.refreshed.connect(self.cleanStatusTextEdit)
        self.CleanWorker.start_clean_signal.connect(self.currentStatusLabel)
        self.CleanWorker.end_clean_signal.connect(self.currentStatusTextEdit)
        self.CleanWorker.seninfo_signal.connect(self.show_senList)
        
        # Thread (Model Prediction)
        self.PredictWorker = PredictWorker()
        thread_modelpred = QThread(self)
        thread_modelpred.start()
        
        self.PredictWorker.moveToThread(thread_modelpred)
        self.predictButton.clicked.connect(self.transfer_testfilecheckState)
        self.predictButton.clicked.connect(self.transfer_modelcheckState)
        self.PredictWorker.refreshed.connect(self.cleanfigureDisplay)
        self.PredictWorker.start_pred_signal.connect(self.showPredLabel)
        self.PredictWorker.prederr_signal.connect(self.showPredictError)
        self.PredictWorker.finished.connect(self.predictSummaryDisplay)
        
        # Thread (Animation)
        self.AnimationWorker = AnimationWorker()
        thread_animation = QThread(self)
        thread_animation.start()
        
        self.AnimationWorker.moveToThread(thread_animation)
        self.AnimationWorker.on_signal.connect(self.receiveAnimationSignal)
        self.AnimationWorker.off_signal.connect(self.receiveAnimationSignal)
                
    def initUI(self): 
        self.viewButton.clicked.connect(self.emitViewSignal)        
        self.browseButton.clicked.connect(self.emitBrowseSignal)
        self.browseSignal.connect(self.show_selfiles)
        
        self.corrButton.clicked.connect(self.corrfigureDisplay)
        
        self.senSelectButton.clicked.connect(self.transfer_checkState)
        self.inoutButton.clicked.connect(self.inout_relation)
        
        self.testchooseButton.clicked.connect(self.emitTestBrowseSignal)
        self.testbrowseSignal.connect(self.show_testselfiles)
        
        self.selmodelButton.clicked.connect(self.emitModelBrowseSignal)
        self.modelbrowseSignal.connect(self.show_modelselfiles)
        
        self.closeButton.clicked.connect(lambda:self.close())
        
        self.showStatus.setReadOnly(True)
        self.showStatus.setAutoFillBackground(False)
        
        self.trainfreezeButton.clicked.connect(self.freezeTraining)
        
        # Set Picture
        pixmap = QPixmap("corr_1.jpg")
        self.corrLabel_1.setPixmap(pixmap)
        pixmap = QPixmap("corr_2.jpg")
        self.corrLabel_2.setPixmap(pixmap)
        pixmap = QPixmap("corr_3.jpg")
        self.corrLabel_3.setPixmap(pixmap)
        pixmap = QPixmap("corr_4.jpg")
        self.corrLabel_4.setPixmap(pixmap)
        pixmap = QPixmap("corr_5.jpg")
        self.corrLabel_5.setPixmap(pixmap)
        
        self.pixmap_1 = QPixmap("Net_1.jpg")        
        self.pixmap_2 = QPixmap("Net_2.jpg")        
        self.pixmap_org = QPixmap("Net.jpg")
        
        QTimer.singleShot(0, \
                      partial(self.AnimationWorker.start_task))
        
        # Set default value for parameters
        self.winlengthEdit.setText('40')
        self.winstepEdit.setText('1')
        self.auto_winlengthEdit.setText('40')
        self.auto_winstepEdit.setText('1')
        
        self.numneuronEdit.setText('50')
        self.epochEdit.setText('100')
        self.filterEdit.setText('36')
        self.modelnameEdit.setText('model')
        self.auto_numneuronEdit.setText('50')
        self.auto_epochEdit.setText('100')
        self.auto_filterEdit.setText('36')        
        
        self.batchszEdit.setText('512')
        self.auto_batchszEdit.setText('512')
        
        # Setting Radio Button
        self.rbtn1.toggled.connect(self.autoselSensor)
        self.rbtn2.toggled.connect(self.autoselSensor)
        
    def receiveAnimationSignal(self, bool):
        inp_sig = bool
        if inp_sig == True:
            self.picLabel.setPixmap(self.pixmap_1)
            self.picLabel_2.setPixmap(self.pixmap_2)
        else:
            self.picLabel.setPixmap(self.pixmap_org)
            self.picLabel_2.setPixmap(self.pixmap_org)

    def emitBrowseSignal(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setFileMode(QFileDialog.ExistingFiles)
        dlg.setFilter( QDir.Files )
        dlg.setNameFilter("Files (*.csv)")
            
        if dlg.exec_():
            self.filenames= dlg.selectedFiles()
            self.browseSignal.emit(self.filenames)
    
    def emitTestBrowseSignal(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setFileMode(QFileDialog.ExistingFiles)
        dlg.setFilter( QDir.Files )
        dlg.setNameFilter("Files (*.csv)")
            
        if dlg.exec_():
            self.testfilenames= dlg.selectedFiles()
            self.testbrowseSignal.emit(self.testfilenames)
    
    def emitModelBrowseSignal(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setFilter( QDir.Files )
        dlg.setNameFilter("Files (*.hdf5)")
            
        if dlg.exec_():
            self.model_filenames= dlg.selectedFiles()
            self.modelbrowseSignal.emit(self.model_filenames)
            
    def emitViewSignal(self):
        if not hasattr(self, 'selectedIndex'):
            msgBox = QMessageBox()
            msgBox.setWindowTitle('錯誤')
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("請先選擇要檢視的檔案")
            msgBox.addButton(QMessageBox.Ok)
            msgBox.exec_()
        else:
            child_win = ChildrenForm(self)
            child_win.show()
        
            # Read csv file
            filenames = self.filenames
            selectedFile = filenames[self.selectedIndex]
        
            sep_idx = selectedFile.rfind('/')
            path = selectedFile[:sep_idx+1]
            name = selectedFile[sep_idx+1:-4]
            selectedFile = path+'/corr_csv/'+name+'_corr.csv'
        
            df = read_csv(selectedFile)
            child_win.open_visualize(self.sensor_name, df)
            
    def currentStatusTextEdit(self,str):
        self.showStatus.append(str)
    
    def cleanStatusTextEdit(self):
        self.showStatus.setText('')
        
    def currentStatusLabel(self,str):
        self.cleanstatusLabel.setText(str)
    
    def visualizeData(self,qModelIndex):
        selectedIndex = qModelIndex.row()
        self.selectedIndex = selectedIndex
        
        # Read csv file
        filenames = self.filenames
        selectedFile = filenames[selectedIndex]
        
        sep_idx = selectedFile.rfind('/')
        path = selectedFile[:sep_idx+1]
        name = selectedFile[sep_idx+1:-4]
        selectedFile = path+'/corr_csv/'+name+'_corr.csv'
        
        df = read_csv(selectedFile)
        
        # Show Excel Content
        model = pandasModel(df)
        self.tableView.setModel(model)
        self.tableView.show()
    
    def show_senList(self, list):
        self.sensor_name = list
        self.checkmodel.setColumnCount(4)
        self.checkmodel.setRowCount(len(self.sensor_name))
        
        horizontalHeader = ["名稱","選擇","最大值","最小值"]
        self.checkmodel.setHorizontalHeaderLabels(horizontalHeader)
        
        if isfile('./norm_parameter.csv'):
            df = read_csv('./norm_parameter.csv')
            column_name = df.columns
            for i in range(len(self.sensor_name)):
                ck = QTableWidgetItem()
                ck.setCheckState(Qt.Unchecked)
                
                self.checkmodel.setItem(i,0,QTableWidgetItem(self.sensor_name[i]))
                self.checkmodel.setItem(i,1,ck)
                
                max_par = int(df[column_name[i]][0])
                min_par = int(df[column_name[i]][1])
                self.checkmodel.setItem(i,2,QTableWidgetItem(str(max_par)))
                self.checkmodel.setItem(i,3,QTableWidgetItem(str(min_par)))
        else:
            for i in range(len(self.sensor_name)):
                ck = QTableWidgetItem()
                ck.setCheckState(Qt.Unchecked)
                
                self.checkmodel.setItem(i,0,QTableWidgetItem(self.sensor_name[i]))
                self.checkmodel.setItem(i,1,ck)
                self.checkmodel.setItem(i,2,QTableWidgetItem(str(0)))
                self.checkmodel.setItem(i,3,QTableWidgetItem(str(0)))
         
        self.checkmodel.resizeColumnsToContents()
        self.checkmodel.resizeRowsToContents()
        self.checkmodel.setColumnWidth(0,85)
        self.checkmodel.setColumnWidth(2,80)
        self.checkmodel.setColumnWidth(3,80)
        
    def transfer_checkState(self):       
        if not hasattr(self, 'sensor_name'):
            msgBox = QMessageBox()
            msgBox.setWindowTitle('錯誤')
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("請先匯入檔案，才能進行感測器確認")
            msgBox.addButton(QMessageBox.Ok)
            msgBox.exec_()
        else:
            sensor_name = self.sensor_name
        
            sensorList = []
            confirmmodel = QStandardItemModel()
            
            # Find check state in sensor list
            for i in range(len(sensor_name)): 
                if self.checkmodel.item(i,1).checkState():
                    item = QStandardItem(str(i+1)+': '+self.checkmodel.item(i,0).text())
                    confirmmodel.appendRow(item)
                    sensorList.append(int(i))
                else:
                    pass
            
            self.confirmmodel = confirmmodel
            self.sensorList = sensorList
        
            self.InputSensorListview.setModel(confirmmodel)
            self.InputSensorListview.show()
    
    def inout_relation(self):
        sensor_name = self.sensor_name
        
        outsensorList = []
        outsensormodel = QStandardItemModel()
        for i in range(len(sensor_name)):
            if self.checkmodel.item(i,1).checkState():
                item = QStandardItem(str(i+1)+': '+self.checkmodel.item(i,0).text())
                outsensormodel.appendRow(item)
                outsensorList.append(int(i))
            else:
                pass
                    
        self.outsensormodel = outsensormodel
        self.OutputSensorListview.setModel(outsensormodel)
        self.OutputSensorListview.show()
    
    def get_targetSensor(self):
        if not hasattr(self, 'outsensormodel'):
            pass
        else:
            sensor_name = self.sensor_name
            print(sensor_name)
            
            sendList = []
            for i in range(len(sensor_name)):
                find_items = self.outsensormodel.findItems(str(i+1)+': '+sensor_name[i])
                if find_items:
                    sendList.append(int(i))            
            self.sendList = sendList
        
    def show_selfiles(self, list):
        model = QStandardItemModel()
        
        for i in range(len(list)):
            show_filename = str(list[i])
            sep_idx = show_filename.rfind('/')
            
            item = QStandardItem(show_filename[sep_idx+1:])
            item.setCheckable(False)
            model.appendRow(item)
        
        self.DataCheckerListview.setModel(model)
        self.DataCheckerListview.show()
        self.DataCheckerListview.clicked.connect(self.visualizeData)
    
    def show_testselfiles(self, list):
        testfile_model = QStandardItemModel()
        
        for i in range(len(list)):
            show_filename = str(list[i])
            sep_idx = show_filename.rfind('/')
            
            item = QStandardItem(show_filename[sep_idx+1:])
            item.setCheckable(True)
            item.setCheckState(Qt.Checked)
            testfile_model.appendRow(item)
        
        self.testfile_model = testfile_model
        self.TestDataListview.setModel(testfile_model)
        self.TestDataListview.show()
        self.TestDataListview.clicked.connect(self.predictfigureDisplay)
        
    def show_modelselfiles(self, list):
        netsel_model = QStandardItemModel()
        
        for i in range(len(list)):
            show_filename = str(list[i])
            sep_idx = show_filename.rfind('/')
            
            item = QStandardItem(show_filename[sep_idx+1:])
            netsel_model.appendRow(item)
        
        self.netsel_model = netsel_model
        self.ModelListview.setModel(netsel_model)
        self.ModelListview.show()
    
    def transfer_modelcheckState(self):        
        QTimer.singleShot(0, \
                      partial(self.PredictWorker.start_task, self.testfileList, self.model_filenames))
    
    def transfer_testfilecheckState(self):
        testfilenames = self.testfilenames
        
        testfileList = []
        for i in range(len(testfilenames)):
            name = testfilenames[i]
            sep_idx = name.rfind('/')
            name = name[sep_idx+1:]
            
            find_items = self.testfile_model.findItems(name)
            if find_items:
                checked = find_items[0].checkState() == Qt.Checked
                if checked == True:
                    testfileList.append(testfilenames[i]) 
        self.testfileList = testfileList
     
    def emitProcessSignal(self):
        if not hasattr(self, 'sendList'):
            msgBox = QMessageBox()
            msgBox.setWindowTitle('錯誤')
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("尚未決定輸入、輸出感測器")
            msgBox.addButton(QMessageBox.Ok)
            msgBox.exec_()
        elif not hasattr(self, 'sensorList'):
            msgBox = QMessageBox()
            msgBox.setWindowTitle('錯誤')
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("尚未決定輸入、輸出感測器")
            msgBox.addButton(QMessageBox.Ok)
            msgBox.exec_()
        else:
            par_name = ['資料流長度', '滾動步幅']
                
            pList = []
            pList.append(self.winlengthEdit.text())
            pList.append(self.winstepEdit.text())
            
            empty_idx = []
            for i in range(len(pList)):
                if pList[i] == '':
                    empty_idx.append(1)
                else:
                    empty_idx.append(0)
            
            if len(self.sendList) == 0:
                msgBox = QMessageBox()
                msgBox.setWindowTitle('錯誤')
                msgBox.setIcon(QMessageBox.Critical)
                msgBox.setText("輸出感測器不能為空")
                msgBox.addButton(QMessageBox.Ok)
                msgBox.exec_()
            elif len(self.sensorList) == 0:
                msgBox = QMessageBox()
                msgBox.setWindowTitle('錯誤')
                msgBox.setIcon(QMessageBox.Critical)
                msgBox.setText("輸入感測器不能為空")
                msgBox.addButton(QMessageBox.Ok)
                msgBox.exec_()
            elif len(self.sendList) > 1:
                msgBox = QMessageBox()
                msgBox.setWindowTitle('錯誤')
                msgBox.setIcon(QMessageBox.Critical)
                msgBox.setText("輸出感測器只能選擇一個")
                msgBox.addButton(QMessageBox.Ok)
                msgBox.exec_()
            elif sum(empty_idx) > 0:
                msgBox = QMessageBox()
                msgBox.setWindowTitle('錯誤')
                msgBox.setIcon(QMessageBox.Critical)
                warning_info = '請輸入'
                for i in range(len(pList)):
                    if empty_idx[i] == 1:
                        warning_info = warning_info+'「'+par_name[i]+'」'
                msgBox.setText(warning_info)
                msgBox.addButton(QMessageBox.Ok)
                msgBox.exec_()
            else:
                pList.append(self.sendList)
                       
                sensorList = self.sensorList
                filenames = self.filenames
                
                max_val, min_val = [], []
                for i in range(self.checkmodel.rowCount()): 
                    max_val.append(float(self.checkmodel.item(i,2).text()))
                    min_val.append(float(self.checkmodel.item(i,3).text()))
                    
                save_info = [range(1,len(max_val)+1), max_val, min_val]
                save_info = DataFrame(save_info)
                save_info.to_csv('norm_parameter.csv', header=False, index=False)
                
                QTimer.singleShot(0, \
                                  partial(self.PreprocessWorker.start_task, pList, sensorList, filenames))
    
    # def save_norminfo(self): 
    #     max_val, min_val = [], []
    #     for i in range(self.checkmodel.rowCount()): 
    #         max_val.append(float(self.checkmodel.item(i,2).text()))
    #         min_val.append(float(self.checkmodel.item(i,3).text()))
            
    #     save_info = [range(1,len(max_val)+1), max_val, min_val]
    #     save_info = DataFrame(save_info)
    #     save_info.to_csv('norm_parameter.csv', header=False, index=False)
        
    def emitCleanSignal(self):
        if not hasattr(self, 'filenames'):
            msgBox = QMessageBox()
            msgBox.setWindowTitle('錯誤')
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("尚未匯入檔案")
            msgBox.addButton(QMessageBox.Ok)
            msgBox.exec_()
        else:        
            filenames = self.filenames
            QTimer.singleShot(0, \
                       partial(self.CleanWorker.start_task, filenames))
    
    def emitAutoSignal(self):
        par_name = ['資料流長度', '滾動步幅']
        
        if not hasattr(self, 'sel_method'):
            msgBox = QMessageBox()
            msgBox.setWindowTitle('錯誤')
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("請設定輸入感測器挑選方式")
            msgBox.addButton(QMessageBox.Ok)
            msgBox.exec_()
        else:
            pList = []
            pList.append(self.winlengthEdit.text())
            pList.append(self.auto_winstepEdit.text())
            pList.append(self.sel_method)
        
        tpList = []
        tpList.append(self.auto_numneuronEdit.text())
        tpList.append(self.auto_epochEdit.text())
        tpList.append(self.auto_filterEdit.text())
        tpList.append(self.winlengthEdit.text())
        tpList.append(self.auto_batchszEdit.text())

        empty_idx = []
        for i in range(len(pList)):
            if pList[i] == '':
                empty_idx.append(1)
            else:
                empty_idx.append(0)
                
        if sum(empty_idx) > 0:
            msgBox = QMessageBox()
            msgBox.setWindowTitle('錯誤')
            msgBox.setIcon(QMessageBox.Critical)
            warning_info = '請輸入'
            for i in range(len(pList)):
                if empty_idx[i] == 1:
                    warning_info = warning_info+'「'+par_name[i]+'」'
            msgBox.setText(warning_info)
            msgBox.addButton(QMessageBox.Ok)
            msgBox.exec_()
        elif not hasattr(self, 'filenames'):
            msgBox = QMessageBox()
            msgBox.setWindowTitle('錯誤')
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("尚未匯入任何檔案")
            msgBox.addButton(QMessageBox.Ok)
            msgBox.exec_()
        else:
            filenames = self.filenames
            QTimer.singleShot(0, partial(self.AutoWorker.start_task, pList, tpList, filenames))
    
    def showProcingTextEdit(self,str):
        self.showStatus_npz.append(str)
        
    def cleanProcingTextEdit(self):
        self.showStatus_npz.setText('')
        
    def showAutoTextEdit(self,str):
        self.showAutoStatus.append(str)
    
    def cleanAutoTextEdit(self):
        self.showAutoStatus.setText('')
        
    def showProcingLabel(self,str):
        self.processstatusLabel.setText(str)
        
    def showAutoLabel(self,str):
        self.autoLabel.setText(str)
        
    def showTrainError(self,str):
        self.finalError_Label.setText(str)
    
    def showTrainTime(self,str):
        self.trainTimeLabel.setText(str)
        
    def showPredictError(self,str):
        self.predictLabel.setText(str)
        
    def showPredLabel(self,str):
        self.predstatusLabel.setText(str)
        
    def emitTrainparSignal(self):
        global stop_train_flag
        stop_train_flag = False
        
        path = './../Data/npz_file/'
        
        tpList = []
        tpList.append(self.numneuronEdit.text())
        tpList.append(self.epochEdit.text())
        tpList.append(self.filterEdit.text())
        tpList.append(self.winlengthEdit.text())
        tpList.append(self.modelnameEdit.text())
        tpList.append(self.outsensorEdit.text())
        tpList.append(self.batchszEdit.text())
        
        empty_idx = []
        for i in range(len(tpList)):
            if tpList[i] == '':
                empty_idx.append(1)
            else:
                empty_idx.append(0)
                
        par_name = ['單層神經元數目','訓練次數','卷積層濾鏡數目','資料流長度','模型名稱','輸出感測器']
        if not exists(path):
            msgBox = QMessageBox()
            msgBox.setWindowTitle('錯誤')
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("不存在轉檔完成的資料")
            msgBox.addButton(QMessageBox.Ok)
            msgBox.exec_()
        elif sum(empty_idx) > 0:
            msgBox = QMessageBox()
            msgBox.setWindowTitle('錯誤')
            msgBox.setIcon(QMessageBox.Critical)
            warning_info = '請輸入'
            for i in range(len(tpList)):
                if empty_idx[i] == 1:
                    warning_info = warning_info+'「'+par_name[i]+'」'
            msgBox.setText(warning_info)
            msgBox.addButton(QMessageBox.Ok)
            msgBox.exec_()
        else:
            train_filenames = [path+_ for _ in listdir(path) if _.endswith(self.outsensorEdit.text()+".npz")]
            if len(train_filenames) == 0:
                msgBox = QMessageBox()
                msgBox.setWindowTitle('錯誤')
                msgBox.setIcon(QMessageBox.Critical)
                msgBox.setText("請確認輸出感測器")
                msgBox.addButton(QMessageBox.Ok)
                msgBox.exec_()
            else:
                self.progressBar.setValue(0)
                QTimer.singleShot(0, \
                                  partial(self.TrainWorker.start_task, train_filenames, tpList))
        
    def corrfigureDisplay(self):   
        self.corrButton.setEnabled(False)
        
        if not hasattr(self, 'filenames'):
            msgBox = QMessageBox()
            msgBox.setWindowTitle('錯誤')
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("請先匯入檔案，才能進行相關性分析")
            msgBox.addButton(QMessageBox.Ok)
            msgBox.exec_()
        else:
            # Read csv file
            filenames = self.filenames
            filenames = filenames[0]
            
            sep_idx = filenames.rfind('/')
            path = filenames[:sep_idx+1]
            
            corr_filenames = [path+'corr_csv/'+_ for _ in listdir(path+'/corr_csv/') if _.endswith("corr.csv")]
            
            all_df = DataFrame()
            for i in corr_filenames:
                df = read_csv(i)            
                all_df = all_df.append(df)
                   
            dr = CorrFigure_Canvas()
            dr.plotData(all_df.iloc[:,1:])
            
            graphicscene = QGraphicsScene()
            graphicscene.addWidget(dr)
            
            self.correlationView.setScene(graphicscene)
            self.correlationView.show()
            
        self.corrButton.setEnabled(True)
        
    def epochfigureDisplay(self):
        train_history = loadmat('train_history.mat')
        train_loss = squeeze(train_history.get('train_loss'))
        val_loss = squeeze(train_history.get('val_loss'))
        losses = squeeze([train_loss, val_loss])
        
        if train_loss.size > 1:
            num_epoch = array(range(1,len(train_loss)+1))
        
            # Visualize Data by Matplotlib
            dr = Epoch_Canvas()
            dr.plotData(num_epoch, losses)
            graphicscene = QGraphicsScene()
            graphicscene.addWidget(dr)
            
            self.epochView.setScene(graphicscene)
            self.epochView.show()
        
    def predictfigureDisplay(self, qModelIndex):
        selectedIndex = qModelIndex.row()
        
        # Read Mat File
        testfilenames = self.testfilenames
        selectedFile = testfilenames[selectedIndex]
        
        sep_idx = selectedFile.rfind('/')
        path = selectedFile[:sep_idx+1]
        name = selectedFile[sep_idx+1:-4]
        selectedFile = path+'/../output_prediction/'+name+'_pred.mat'
        
        mat = loadmat(selectedFile)
        pred_result = squeeze(mat.get('pred_result'))
        target = squeeze(mat.get('target'))
        residual = squeeze(mat.get('residual'))
        x_axis = squeeze(mat.get('time_axis'))
        
        # Visualize Data by Matplotlib
        dr = Predict_Canvas()
        dr.plotData(x_axis, pred_result, 'Prediction')
        dr.plotData(x_axis, target, 'True')
        graphicscene = QGraphicsScene()
        graphicscene.addWidget(dr)
        
        self.signalView.setScene(graphicscene)
        self.signalView.show()
        
        # Visualize Residual
        dr = Predict_Canvas()
        dr.plotData(x_axis, residual, 'Residual')
        graphicscene = QGraphicsScene()
        graphicscene.addWidget(dr)
        
        self.residualView.setScene(graphicscene)
        self.residualView.show()
        
    def predictSummaryDisplay(self):
        testfilenames = self.testfilenames
        testfilenames = testfilenames[0]
        
        sep_idx = testfilenames.rfind('/')
        path = testfilenames[:sep_idx+1]+'/../output_prediction/'
        
        matfile = [_ for _ in listdir(path) if _.endswith(".mat")]
        
        total_target = empty([0])
        total_pred = empty([0])
        for i in range(len(matfile)):
            mat = loadmat(path+matfile[i])
            pred_result = squeeze(mat.get('pred_result'))
            target = squeeze(mat.get('target'))
            
            total_target = append(total_target, target, axis = 0)
            total_pred = append(total_pred, pred_result, axis = 0)
            
        dr = PredictSum_Canvas()
        dr.plotData(total_pred, 'Prediction')
        dr.plotData(total_target, 'True')
        graphicscene = QGraphicsScene()
        graphicscene.addWidget(dr)
        
        self.totalsignalView.setScene(graphicscene)
        self.totalsignalView.show()
        
    def cleanfigureDisplay(self):
        dr = Predict_Canvas()
        dr.setting()
        graphicscene = QGraphicsScene()
        graphicscene.addWidget(dr)
        self.signalView.setScene(graphicscene)
        self.signalView.show()
        self.residualView.setScene(graphicscene)
        self.residualView.show()
        
        dr = PredictSum_Canvas()
        dr.setting()
        graphicscene = QGraphicsScene()
        graphicscene.addWidget(dr)
        self.totalsignalView.setScene(graphicscene)
        self.totalsignalView.show()
        
    def showProgress(self, current_epoch, error_lists):
        max_epoch = self.epochEdit.text()
        max_epoch = int(max_epoch)
        epoch_ratio = round(((current_epoch+1)/max_epoch)*100)
        
        if current_epoch < 100:
            self.progressBar.setValue(epoch_ratio)
            
    def freezeTraining(self):
        global stop_train_flag
        stop_train_flag = True
        
    def autoselSensor(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            sel_method = radioBtn.text()
            if sel_method == '全部選擇':
                self.sel_method = 0
            elif sel_method == '依相關性係數選擇':
                self.sel_method = 1
                
class ChildrenForm(QMainWindow, Ui_ChildrenForm):
    def __init__(self, parent = None):
        QMainWindow.__init__(self, parent)
        self.ui = Ui_ChildrenForm()
        self.ui.setupUi(self)
    
    def open_visualize(self, sensor_name, df):
        self.ui.visualizeData(sensor_name, df)
              
if __name__=="__main__":  
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication(argv)

    myWin = MyMainWindow()  
    myWin.show()  
    exit(app.exec_())  

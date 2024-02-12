import os
import tkinter as tk
import tkinter.messagebox as messagebox
from tkinter import ttk
from tkinter import Label
import pandas as pd
import numpy as np
from tkinter import Tk, Toplevel, Label, Entry, Button, filedialog, Canvas
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numpy.fft import fft, ifft
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import turtle
import tensorflow as tf
import math
from tkinter import PhotoImage
from PIL import Image, ImageTk

import keras
from keras.optimizers import SGD,Adagrad,RMSprop,Adam,Adadelta,Adamax,Adafactor,AdamW,Nadam
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization,LayerNormalization,GroupNormalization
from sklearn.preprocessing import MinMaxScaler
from matplotlib.figure import Figure




# Function to make a window scrollable
# def on_mousewheel(event):
#     canvas.yview_scroll(-1 * (event.delta // 120), "units")

# def make_scrollable(window):
#     global canvas
    
#     canvas = tk.Canvas(window)
#     scrollbar = tk.Scrollbar(window, orient="vertical", command=canvas.yview)
#     frame = tk.Frame(canvas)

#     frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
#     canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
#     canvas.bind_all("<MouseWheel>", on_mousewheel)

#     canvas.create_window((0, 0), window=frame, anchor="nw")
#     canvas.configure(yscrollcommand=scrollbar.set)

#     return canvas, frame, scrollbar




# Our main window with the entered name of the project
def next_window():
    root1 = Toplevel(root)
    root1.title(entry_name.get())
    root1.state("zoomed")

    # Make the root1 window scrollable
    # canvas, frame, scrollbar = make_scrollable(root1)


# Data prepration
# HC,UB,CR,BPFO,BPFI

    def process_data(directory, fault_names, speeds, domain):
        X = []
        Y = []

        print(fault_names)
        for f in range(len(fault_names)):
        
            for s in speeds:
                os.chdir(directory)  # used to change the current wd to the given directory
                c = os.getcwd()  # c is the current working directory
                P = os.path.join(c, fault_names[f])  # added fault[f] as a seperate directory in c i.e the current working directory
                S = str(s) + 'Hz'
                P = os.path.join(P, S)  # added a dircetory named '40Hz' inside the directory named 'fault[f]'
                P = os.path.join(P,'filtered_timedomain')  # added a directory inside the 40Hz directory named as 'filtered_timedomain

                for i in range(int(entry_datasets.get())):
                    file = ''
                    file = fault_names[f] + '_time_' + S + '_' + str(i) + '.xls'
                    final_file = os.path.join(P, file)
                    print(final_file)
                    df = pd.read_csv(final_file, sep='\t')
                    df.columns = range(1+int(entry_sensors.get()))  #no. of sensors + 1 coloumn for Time coordinate
                    df = df.iloc[:, 1:]                 #take from 6columns after excluding the first column
                     #created a df having columns [1,2,3,4,5,6...,no. of sensors]

                    if domain.lower() == 'time':
                        layers = 1
                        le = 4096
                        for i in range(math.floor(int(entry_points.get())/4096)):
                            start_at = i * le
                            end_at = i * le + le
                            df2 = df.iloc[start_at:end_at, :]
                            X.append(df2)
                            Y.append(f)
                    else:
                        layers = 1
                        le = 4096

                        df4 = np.zeros((2048, 6))
                        for i in range(200):
                            for j in range(6):
                                start_at = i * le
                                end_at = i * le + le
                                df3 = df.iloc[start_at:end_at, j]
                                DF3 = fft(df3)
                                DF3 = np.abs(DF3)
                                DF3 = DF3[0:2048]
                                DF3 = DF3.tolist()
                                df4[0:2048, j] = DF3[0:2048]
                            df5 = pd.DataFrame(df4)
                            X.append(df5)
                            Y.append(f)

        X = np.array(X)
        Y = np.array(Y)     #f=fault[f]
        print(len(X),len(Y))
        if domain.lower() == 'time':
            X = X.reshape((len(X), 4096, int(entry_sensors.get()), 1))  #condition for domain le or le2
        elif domain.lower() == 'fft':
            X = X.reshape((len(X), 2048, 6, 1))
        else:
            pass

        X, Y = shuffle(X, Y)
        return X,Y
    
    # Function to add feature of testing the model on any other file
   
                           
    def split(test_size, validation_size,X,Y):
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=float(test_size),shuffle=True)  # splitted the data into train and test
        X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=float(validation_size),shuffle=True)  # splitted the train data into train and validation data
        X_train = np.array(X_train)
        X_validation = np.array(X_validation)
        Y_train = np.array(Y_train)
        Y_validation = np.array(Y_validation)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)

        return X_train, Y_train, X_test, Y_test, X_validation, Y_validation

    

    class ConvolutionalLayerInput:
        def __init__(self, layer_number, root):
            self.layer_number = layer_number
            self.root = root
            # self.frame = ttk.Frame(self.root)
            self.frame = ttk.Frame(self.root)
            

            self.labels = []
            self.entries = []
            
            
            self.activation_var=tk.StringVar()
            self.kernel_var=tk.StringVar()
            self.pad_var=tk.StringVar()
            self.norm_var=tk.StringVar()
            self.pool_var=tk.StringVar()
            
            
            def on_select1(event):
                activation_value=self.activation_var.get()
                kernel_value=self.kernel_var.get()
                pad_value=self.pad_var.get()
                norm_value=self.norm_var.get()
                pool_value=self.pool_var.get()
            
            if self.layer_number==1:
                input_options = ["Number of Kernels:", "Kernel Size(in brackets):", "Strides:", "Activation:", "Kernel Initializer:", "Padding:", "Normalization:", "Pooling:","Pool size(in brackets):"]
                for i, option in enumerate(input_options):
                    
                    if option=="Activation:":
                        
                        activation_label = ttk.Label(self.frame, text=f"{option}")
                        activation_label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
                        options = ['relu','softmax','Sigmoid','Tanh','Leaky ReLU']
                        combobox_activation = ttk.Combobox(self.frame,textvariable=self.activation_var, values=options)
                        combobox_activation.grid(row=i, column=1, padx=5, pady=5)
                        combobox_activation.bind("<<ComboboxSelected>>", on_select1)
                        self.labels.append(activation_label)
                        self.entries.append(self.activation_var)
                    
                    elif option=="Kernel Initializer:":
                        
                        kernel_label = ttk.Label(self.frame, text=f"{option}")
                        kernel_label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
                        options = ['random_normal']
                        combobox_kernel = ttk.Combobox(self.frame,textvariable=self.kernel_var, values=options)
                        combobox_kernel.grid(row=i, column=1, padx=5, pady=5)
                        combobox_kernel.bind("<<ComboboxSelected>>", on_select1)
                        self.labels.append(kernel_label)
                        self.entries.append(self.kernel_var)
                        
                    elif option=="Padding:":
                        
                        pad_label = ttk.Label(self.frame, text=f"{option}")
                        pad_label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
                        options = ['same']
                        combobox_pad = ttk.Combobox(self.frame,textvariable=self.pad_var, values=options)
                        combobox_pad.grid(row=i, column=1, padx=5, pady=5)
                        combobox_pad.bind("<<ComboboxSelected>>", on_select1)
                        self.labels.append(pad_label)
                        self.entries.append(self.pad_var)
                    
                    elif option=="Normalization:":
                        
                        norm_label = ttk.Label(self.frame, text=f"{option}")
                        norm_label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
                        options = ['Batch Normalization', 'Layer Normalization', 'Group Normalization']
                        combobox_norm = ttk.Combobox(self.frame,textvariable=self.norm_var, values=options)
                        combobox_norm.grid(row=i, column=1, padx=5, pady=5)
                        combobox_norm.bind("<<ComboboxSelected>>", on_select1)
                        self.labels.append(norm_label)
                        self.entries.append(self.norm_var)
                    
                    elif option=="Pooling:":
                        
                        pool_label = ttk.Label(self.frame, text=f"{option}")
                        pool_label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
                        options = ['MaxPooling2D','AveragePooling2D']
                        combobox_pool = ttk.Combobox(self.frame,textvariable=self.pool_var, values=options)
                        combobox_pool.grid(row=i, column=1, padx=5, pady=5)
                        combobox_pool.bind("<<ComboboxSelected>>", on_select1)
                        self.labels.append(pool_label)
                        self.entries.append(self.pool_var)
                    
                    else:
                        
                        label = ttk.Label(self.frame, text=f"{option}")
                        entry = ttk.Entry(self.frame)
                        label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
                        entry.grid(row=i, column=1, padx=5, pady=5)
                        self.labels.append(label)
                        self.entries.append(entry)
                
                
                
                
                
                # for i, option in enumerate(input_options):
                #     label = ttk.Label(self.frame, text=f"{option}")
                #     entry = ttk.Entry(self.frame)
                #     label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
                #     entry.grid(row=i, column=1, padx=5, pady=5)
                #     self.labels.append(label)
                #     self.entries.append(entry)
            
            else:
                input_options = ["Number of Kernels:", "Kernel Size(in brackets):", "Strides:", "Kernel Initializer:", "Padding:", "Normalization:", "Pooling:","Pool size(in brackets):"]
                for i, option in enumerate(input_options):
                    if option=="Kernel Initializer:":
                        
                        kernel_label = ttk.Label(self.frame, text=f"{option}")
                        kernel_label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
                        options = ['random_normal']
                        combobox_kernel = ttk.Combobox(self.frame,textvariable=self.kernel_var, values=options)
                        combobox_kernel.grid(row=i, column=1, padx=5, pady=5)
                        combobox_kernel.bind("<<ComboboxSelected>>", on_select1)
                        self.labels.append(kernel_label)
                        self.entries.append(self.kernel_var)
                        
                    elif option=="Padding:":
                        
                        pad_label = ttk.Label(self.frame, text=f"{option}")
                        pad_label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
                        options = ['same']
                        combobox_pad = ttk.Combobox(self.frame,textvariable=self.pad_var, values=options)
                        combobox_pad.grid(row=i, column=1, padx=5, pady=5)
                        combobox_pad.bind("<<ComboboxSelected>>", on_select1)
                        self.labels.append(pad_label)
                        self.entries.append(self.pad_var)
                    
                    elif option=="Normalization:":
                        
                        norm_label = ttk.Label(self.frame, text=f"{option}")
                        norm_label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
                        options = ['Batch Normalization', 'Layer Normalization', 'Group Normalization']
                        combobox_norm = ttk.Combobox(self.frame,textvariable=self.norm_var, values=options)
                        combobox_norm.grid(row=i, column=1, padx=5, pady=5)
                        combobox_norm.bind("<<ComboboxSelected>>", on_select1)
                        self.labels.append(norm_label)
                        self.entries.append(self.norm_var)
                    
                    elif option=="Pooling:":
                        
                        pool_label = ttk.Label(self.frame, text=f"{option}")
                        pool_label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
                        options = ['MaxPooling2D','AveragePooling2D']
                        combobox_pool = ttk.Combobox(self.frame,textvariable=self.pool_var, values=options)
                        combobox_pool.grid(row=i, column=1, padx=5, pady=5)
                        combobox_pool.bind("<<ComboboxSelected>>", on_select1)
                        self.labels.append(pool_label)
                        self.entries.append(self.pool_var)
                    
                    else:
                        
                        label = ttk.Label(self.frame, text=f"{option}")
                        entry = ttk.Entry(self.frame)
                        label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
                        entry.grid(row=i, column=1, padx=5, pady=5)
                        self.labels.append(label)
                        self.entries.append(entry)
                
            
            # Disable activation entry for layers other than the first
            # if self.layer_number != 1:
            #     self.entries[3].configure(state='disabled')
            #     self.labels[3].configure(state='disabled')

        def add_layer(self,model,input_shape=(0,0,0)):
            
            if input_shape==(0,0,0) or self.layer_number!=1:
                
                model.add(Conv2D(int(self.entries[0].get()), kernel_size=eval(self.entries[1].get()), strides=eval(self.entries[2].get()), kernel_initializer=self.kernel_var.get(), padding=self.pad_var.get()))
                
                if self.entries[-3]=='Batch Normalization':
                    model.add(BatchNormalization())
                elif self.entries[-3]=='Layer Normalization':
                    model.add(LayerNormalization())
                elif self.entries[-3]=='Group Normalization':
                    model.add(GroupNormalization())
                    
                if self.entries[-2]=='MaxPooling2D':
                    model.add(MaxPooling2D(self.entries[-1]))
                elif self.entries[-2]=='AveragePooling2D':
                    model.add(AveragePooling2D(self.entries[-1]))
                    
            else:
                
                model.add(Conv2D(int(self.entries[0].get()), kernel_size=eval(self.entries[1].get()), strides=eval(self.entries[2].get()), activation=self.activation_var.get(), kernel_initializer=self.kernel_var.get(), padding=self.pad_var.get(),input_shape=input_shape))
                
                if self.entries[-3]=='Batch Normalization':
                    model.add(BatchNormalization())
                elif self.entries[-3]=='Layer Normalization':
                    model.add(LayerNormalization())
                elif self.entries[-3]=='Group Normalization':
                    model.add(GroupNormalization())
                    
                if self.entries[-2]=='MaxPooling2D':
                    model.add(MaxPooling2D(self.entries[-1]))
                elif self.entries[-2]=='AveragePooling2D':
                    model.add(AveragePooling2D(self.entries[-1]))                
                
    # def build_model(input_shape, fault_names, normalization, pooling, dense_activation, optimizer, metric):
    def build_model(fault_names, dense_activation, optimizer_value, metric_value, input_shape=(0,0,0)):
        
        model = Sequential()
        # # Layer1
        # model.add(Conv2D(8, kernel_size=(4, 2), strides=(1, 1), activation='relu', kernel_initializer='random_normal',padding="same", input_shape=input_shape))
        # # model.add(Conv2D(8,kernel_size=(4,2),strides=(1,1),activation='relu',kernel_initializer='random_normal',input_shape=input_shape))
        # if normalization=='Batch Normalization':
        #     model.add(BatchNormalization())
        # elif normalization=='Layer Normalization':
        #     model.add(LayerNormalization())
        # elif normalization=='Group Normalization':
        #     model.add(GroupNormalization())
            
            
        # if pooling=='MaxPooling2D':
        #     model.add(MaxPooling2D(pool_size=(4,1)))
        # elif pooling=='AveragePooling2D':
        #     model.add(AveragePooling2D(pool_size=(4,1)))

        # # Layer2
        # model.add(Conv2D(16, kernel_size=(8, 2), strides=(8, 1), kernel_initializer='random_normal', padding="same"))
        # # model.add(Conv2D(16,kernel_size=(8,2),strides=(8,1),kernel_initializer='random_normal'))
        # if normalization=='Batch Normalization':
        #     model.add(BatchNormalization())
        # elif normalization=='Layer Normalization':
        #     model.add(LayerNormalization())
        # elif normalization=='Group Normalization':
        #     model.add(GroupNormalization())


        # if pooling=='MaxPooling2D':
        #     model.add(MaxPooling2D(pool_size=(4,1)))
        # elif pooling=='AveragePooling2D':
        #     model.add(AveragePooling2D(pool_size=(4,1)))

        # # Layer3
        # model.add(Conv2D(32, kernel_size=(16, 2), strides=(8, 2), kernel_initializer='random_normal', padding="same"))
        # if normalization=='Batch Normalization':
        #     model.add(BatchNormalization())
        # elif normalization=='Layer Normalization':
        #     model.add(LayerNormalization())
        # elif normalization=='Group Normalization':
        #     model.add(GroupNormalization())


        # if pooling=='MaxPooling2D':
        #     model.add(MaxPooling2D(pool_size=(4,1)))
        # elif pooling=='AveragePooling2D':
        #     model.add(AveragePooling2D(pool_size=(4,1)))
        
        
        Layer_no=1
        
        while ( Layer_no < int(entry_layers.get()) ) :
            if Layer_no==1:
                convolutional_layers[0].add_layer(model,input_shape)
            else:
                convolutional_layers[Layer_no-1].add_layer(model)
                
            Layer_no+=1

        # Flattening the conv Layer and adding dense Layers
        model.add(keras.layers.Flatten())
        # model.add(keras.layers.Dense(500, activation='relu',kernel_initializer='random_normal'))
        model.add(keras.layers.Dense(100, activation='relu', kernel_initializer='random_normal'))
        model.add(Dense(len(fault_names), activation=dense_activation))
        return model


    def get_directory():
        directory = filedialog.askdirectory()
        entry_directory.delete(0, "end")
        entry_directory.insert(0, directory)

    def process_button_clicked():
        
        global save_label
        save_label.grid_forget()
        metric_value = metric_var.get()
        optimizer_value = optimizer_var.get()
        # pooling_value = pooling_var.get()
        activation_value = activation_var.get()
        # norm_value = norm_var.get()
        
        
        directory = entry_directory.get()
        fault_names = entry_fault.get().split(',')                             #gets the data for path, fault names, speeds, domain type
        speeds = [int(speed) for speed in entry_speed.get().split(',')]
        domain = entry_domain.get()

        X,Y=process_data(directory,fault_names,speeds,domain)
        X_train, Y_train, X_test, Y_test, X_validation, Y_validation=split(entry_testFraction.get(),entry_validFraction.get(),X,Y)
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
        # model=build_model(input_shape, fault_names, norm_value, pooling_value, activation_value, optimizer_value, metric_value)
        model=build_model(fault_names, activation_value, optimizer_value, metric_value, input_shape)
        # SGD, Adagrad, RMSprop, Adam, Adadelta, Adamax, Adafactor, AdamW, Nadam
        if optimizer_value=='SGD':
            optimizer=SGD(learning_rate=0.001)
        elif optimizer_value=='Adagrad':
            optimizer=Adagrad(learning_rate=0.001)
        elif optimizer_value=='RMSprop':
            optimizer=RMSprop(learning_rate=0.001)
        elif optimizer_value=='Adam':
            optimizer=Adam(learning_rate=0.001)
        elif optimizer_value=='Adadelta':
            optimizer=Adadelta(learning_rate=0.001)
        elif optimizer_value=='Adamax':
            optimizer=Adamax(learning_rate=0.001)
        elif optimizer_value=='AdamW':
            optimizer=AdamW(learning_rate=0.001)
        elif optimizer_value=='Adafactor':
            optimizer=Adafactor(learning_rate=0.001)
        elif optimizer_value=='Nadam':
            optimizer=Nadam(learning_rate=0.001)


        model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=[metric_value])
        model.summary()

        # int64_tensor_epochs = tf.cast(entry_epochs.get(), dtype=tf.int64)
        # int64_tensor_batchsize=tf.cast(entry_batch_size.get(),dtype=tf.int64)
        batchsize=entry_batch_size.get()
        batchsize_new=int(batchsize)
        
        epochs=entry_epochs.get()
        epochs_new=int(epochs)
        
        history = model.fit(X_train, Y_train, batch_size=batchsize_new, epochs=epochs_new, verbose=1,validation_data=(X_validation, Y_validation), shuffle=True)
                
        train_loss = history.history['loss']
        
        epochs = range(1, len(train_loss) + 1)
        
        validation_loss = history.history['val_loss']
        
        train_accuracy = history.history['accuracy']
        
        validation_accuracy = history.history['val_accuracy']
        
        test_accuracy = model.evaluate(X_test, Y_test)
        

        # canvas_frame.destroy()
        
        
        # Find seperately the validation and test score and then print it
        # Test Accuracy
        Label(root1, text="Test Accuracy:").place(x=700,y=400)
        
        test_accuracy_value=tk.DoubleVar()
        test_accuracy_value.set(test_accuracy[1])
        
        entry_test_accuracy = Entry(root1, width=15, textvariable=test_accuracy_value)
        entry_test_accuracy.place(x=800,y=400)
        
        # Validation Score
        Label(root1, text="Validation Score:").place(x=950,y=400)
        
        val_accuracy_value = tk.DoubleVar()
        val_accuracy_value.set(validation_accuracy[-1])
        
        entry_validation_score = Entry(root1, width=15,textvariable=val_accuracy_value)
        entry_validation_score.place(x=1050,y=400)

        fig_1 = Figure(figsize=(3.5,4.0), facecolor="#917FB3")
        ax_1 = fig_1.add_subplot()
        ax_1.set_facecolor("#917FB3")
        ax_1.set_title('LOSS PLOT')


        # ax_1.fill_between(x=epochs, y1=train_loss, alpha=0.7)
        ax_1.plot(epochs, train_loss, color="deepskyblue")
        ax_1.plot(epochs,validation_loss,color="blue")
        ax_1.legend(['Train loss','Validation loss'])
        ax_1.grid(visible=True)
        
        canvas1 = FigureCanvasTkAgg(figure=fig_1, master=root1)
        canvas1.draw()
        canvas1.get_tk_widget().place(x=700, y=70)


        fig_2 = Figure(figsize=(3.5,4.0), facecolor="#917FB3")
        ax_2 = fig_2.add_subplot()
        ax_2.set_facecolor("#917FB3")
        ax_2.set_title('ACCURACY PLOT')
        # ax_2.fill_between(x=epochs, y1=train_accuracy, alpha=0.7)
        ax_2.plot(epochs, train_accuracy, color="deepskyblue")
        ax_2.plot(epochs, validation_accuracy, color="blue")
        ax_2.legend(['Train accuracy','Validation accuracy'])
        ax_2.grid(visible=True)
        
        canvas2 = FigureCanvasTkAgg(figure=fig_2, master=root1)
        canvas2.draw()
        canvas2.get_tk_widget().place(x=1000, y=70)
        
        

        def test_model(file_path, model):
            df_test = pd.read_csv(file_path, sep='\t')
            df_test.columns = range(1+int(entry_sensors.get()))
            df_test = df_test.iloc[:, 1:]
    
            if domain.lower() == 'time':
                le = 4096
                X_test = []
    
                for i in range(math.floor(int(entry_points.get())/4096)):
                    start_at = i * le
                    end_at = i * le + le
                    df2 = df_test.iloc[start_at:end_at, :]
                    X_test.append(df2)
    
                X_test = np.array(X_test).reshape((len(X_test), 4096, int(entry_sensors.get()), 1))
    
            elif domain.lower() == 'fft':
                le = 4096
                df4 = np.zeros((2048, 6))
    
                for i in range(200):
                    for j in range(6):
                        start_at = i * le
                        end_at = i * le + le
                        df3 = df_test.iloc[start_at:end_at, j]
                        DF3 = fft(df3)
                        DF3 = np.abs(DF3)
                        DF3 = DF3[0:2048]
                        DF3 = DF3.tolist()
                        df4[0:2048, j] = DF3[0:2048]
    
                    df5 = pd.DataFrame(df4)
                    X_test.append(df5)
    
                X_test = np.array(X_test).reshape((len(X_test), 2048, 6, 1))
    
            else:
                return  # handle the case when the domain is not recognized
    
            Y_test_pred = model.predict(X_test)
            # Process Y_test_pred as needed, e.g., convert to class labels
    
            # print(Y_test_pred)
            return Y_test_pred
    
    
        def browse_test_file():
            test_file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xls;*.xlsx")])
            entry_test_file.delete(0, "end")
            entry_test_file.insert(0, test_file_path)
    
        def test_button_clicked():
            
            # fault_name=['HC','UB','CR','BPFO','BPFI']
            user_input = entry_fault.get()
            faults=[name.strip() for name in user_input.split(',')]
            
            test_file_path = entry_test_file.get()
            if not test_file_path:
                messagebox.showwarning("Warning", "Please select a test file.")
                return
    
            Y_test_pred = test_model(test_file_path, model) # Further processing of Y_test_pred, e.g., display or use the results
            
            # Extract list of columns with largest values from each row
            max_columns = np.argmax(Y_test_pred, axis=1)
            
            # Obtain the column number with the most occurrences
            fault_no = np.bincount(max_columns).argmax()
            
            Label(root1, text="Predicted Fault Type:").place(x=710,y=500)     
            fault_type=tk.StringVar()
            fault_type.set(faults[fault_no])
            
            entry_fault_type=Entry(root1,textvariable=fault_type)
            entry_fault_type.place(x=850,y=500)
            
            
        Button(root1, text="TEST for another data", command=test_button_clicked).place(x=700,y=460)
        
        entry_test_file = Entry(root1, width=30)
        entry_test_file.place(x=850,y=463)
        
        Button(root1,text="Browse",command=browse_test_file).place(x=1050,y=460)
    
        
    Label(root1, text="No. of sensors:").grid(row=0, column=0, padx=10, pady=5)
    entry_sensors = Entry(root1, width=5)
    entry_sensors.grid(row=0, column=1, padx=10, pady=5)
    
    Label(root1, text="No. of datasets per class per sensor:").grid(row=1, column=0, padx=10, pady=5)
    entry_datasets = Entry(root1, width=5)
    entry_datasets.grid(row=1, column=1, padx=10, pady=5)
    
    Label(root1, text="No. of points in each dataset:").grid(row=2, column=0, padx=10, pady=5)
    entry_points = Entry(root1, width=5)
    entry_points.grid(row=2, column=1, padx=10, pady=5)

    Label(root1, text="Fault Names (comma-separated):").grid(row=3, column=0, padx=10, pady=5)
    entry_fault = Entry(root1, width=30)
    entry_fault.grid(row=3, column=1, padx=10, pady=5)
    
    Label(root1, text="Speeds:").grid(row=4, column=0, padx=10, pady=5)
    entry_speed = Entry(root1, width=15)
    entry_speed.grid(row=4, column=1, padx=10, pady=5)
    
    Label(root1, text="Type Domain(time/frequency):").grid(row=5, column=0, padx=10, pady=5)
    entry_domain = Entry(root1, width=10)
    entry_domain.grid(row=5, column=1, padx=10, pady=5)
    
    Label(root1, text="Select Input File:").grid(row=6, column=0, padx=0, pady=5)
    entry_directory = Entry(root1, width=50)
    entry_directory.grid(row=6, column=1, padx=10, pady=5)

    Button(root1, text="Browse", command=get_directory).grid(row=6, column=2, padx=10, pady=5)
    
    Label(root1, text="Splitting the data:", font=("Helvetica", 10)).grid(row=7, column=0, columnspan=2, padx=(0, 0), pady=5)
    
    Label(root1, text="Fraction to be used as test data:").grid(row=8, column=0, padx=10, pady=5)
    entry_testFraction = Entry(root1, width=5)
    entry_testFraction.grid(row=8, column=1, padx=10, pady=5)
    
    Label(root1, text="Fraction to be used for validation:").grid(row=9, column=0, padx=10, pady=5)
    entry_validFraction = Entry(root1, width=5)
    entry_validFraction.grid(row=9, column=1, padx=10, pady=5)
    
    Label(root1, text="Optimizing parameters:", font=("Helvetica", 10)).grid(row=10, column=0, columnspan=2, padx=(0, 0), pady=5)
    
    Label(root1, text="Batch Size:").grid(row=11, column=0, padx=10, pady=5)
    entry_batch_size = Entry(root1, width=5)
    entry_batch_size.grid(row=11, column=1, padx=10, pady=5)
    
    Label(root1, text="No. of Epochs:").grid(row=12, column=0, padx=10, pady=5)
    entry_epochs = Entry(root1, width=5)
    entry_epochs.grid(row=12, column=1, padx=10, pady=5)
    
    
    # Setting an event listener for when an option is selected
    def on_select(event):
        # Perform actions based on the selected option
        metric_value=metric_var.get()
        optimizer_value=optimizer_var.get()
        # pooling_value=pooling_var.get()
        # activation_value=activation_var.get()
        # norm_value=norm_var.get()
        
    
    # Creating a Combobox with options for parameters that can be selected by the user instead of being typed fully
    ttk.Label(root1, text="Select the metric to check performance:").grid(row=13, column=0, padx=10, pady=5)
    metric_var=tk.StringVar()
    options = ['accuracy', 'F1-score']
    combobox_metric = ttk.Combobox(root1,textvariable=metric_var, values=options)
    combobox_metric.grid(row=13, column=1, padx=10, pady=5)
    combobox_metric.bind("<<ComboboxSelected>>", on_select)
    
    ttk.Label(root1, text="Select an optimizer:").grid(row=14, column=0, padx=10, pady=5)
    optimizer_var=tk.StringVar()
    options = ['SGD','Adagrad','RMSprop','Adam','Adadelta','Adamax','AdamW','Adafactor','Nadam']
    combobox_optimizer = ttk.Combobox(root1,textvariable=optimizer_var, values=options)
    combobox_optimizer.grid(row=14, column=1, padx=10, pady=5)
    combobox_optimizer.bind("<<ComboboxSelected>>", on_select)

    # Label(root1, text="Type of pooling:").grid(row=14, column=0, padx=10, pady=5)
    # pooling_var=tk.StringVar()
    # options = ['MaxPooling2D','AveragePooling2D']
    # combobox_pooling = ttk.Combobox(root1,textvariable=pooling_var, values=options)
    # combobox_pooling.grid(row=14, column=1, padx=10, pady=5)
    # combobox_pooling.bind("<<ComboboxSelected>>", on_select)
    
    Label(root1, text="Select Activation Function:").grid(row=15, column=0, padx=10, pady=5)
    activation_var=tk.StringVar()
    options = ['softmax','Linear','Sigmoid','Tanh','relu','Leaky ReLU']
    combobox_activation = ttk.Combobox(root1,textvariable=activation_var, values=options)
    combobox_activation.grid(row=15, column=1, padx=10, pady=5)
    combobox_activation.bind("<<ComboboxSelected>>", on_select)
    
    # ttk.Label(root1, text="Type of Normalization:").grid(row=16, column=0, padx=10, pady=10)
    # norm_var=tk.StringVar()
    # options = ['Batch Normalization', 'Layer Normalization', 'Group Normalization']
    # combobox_norm = ttk.Combobox(root1,textvariable=norm_var, values=options)
    # combobox_norm.grid(row=16, column=1, padx=10, pady=5)
    # combobox_norm.bind("<<ComboboxSelected>>", on_select)

    
    Button(root1, text="TRAIN THE MODEL", command=process_button_clicked, width=100).grid(row=16, column=1, columnspan=8, padx=(30, 0), pady=10)
    

    # canvas.pack(side="left", fill="both", expand=True)
    # scrollbar.pack(side="right", fill="y")
    
    
    arch_label = Label(root1, text="Network Architecture Design", font=("Helvetica", 10))
    arch_label.grid(row=0, column=5, columnspan=6, padx=(200, 0), pady=10)
    label_layers = Label(root1, text="Number of Convolutional Layers:")
    entry_layers = Entry(root1)

    canvas_frame = ttk.Frame(root1)
    
    # Add a label indicating the layer number
    layer_label = Label(root1)
    layer_label.grid_forget()
    
    label_layers.grid(row=1, column=3, columnspan=6, padx=(50, 0), pady=10)
    entry_layers.grid(row=1, column=5, columnspan=6, padx=(250, 0), pady=10)
    canvas_frame.place(x=800, y=75)
    
    
    global convolutional_layers
    convolutional_layers = []
    global current_page
    current_page=0
    global num_layers
    num_layers=0
    

    def layer_details():
        global num_layers
        global convolutional_layers
        num_layers = int(entry_layers.get())

        for i in range(1, num_layers + 1):
            convolutional_layer = ConvolutionalLayerInput(i, canvas_frame)
            convolutional_layer.frame.grid(row=0, column=0, columnspan=2, pady=10, sticky="ne")
            convolutional_layers.append(convolutional_layer)
            
            
        def show_page():
            global current_page
            global convolutional_layers
            for i, layer in enumerate(convolutional_layers):
                layer.frame.grid_forget()
                convolutional_layers[current_page].frame.grid(row=0, column=0, columnspan=2, pady=10, sticky="nsew")
    
            label_layers.config(text=f"Page {current_page + 1} of {num_layers}")
            
            layer_label.config(text=f"Layer {current_page+1}",font=("Helvetica", 10))
            layer_label.grid(row=1, column=6, columnspan=6, padx=(450, 0), pady=10)
            
            
        show_page()
        button_layer_details.grid_forget()
        entry_layers.grid_forget()


    
        def next_page():
            global current_page
            if current_page < num_layers - 1:
                current_page += 1
                show_page()
    
        def previous_page():
            global current_page
            if current_page > 0:
                current_page -= 1
                show_page()
    
        def save_configuration(arch_label,layer_label,label_layers):
            # Add your saving logic here
            global convolutional_layers
            save_label.config(text="Configuration Saved!")
            save_label.grid(row=8, column=6, columnspan=6, padx=(100, 0), pady=10) 
            forward_button.grid_forget()
            forward_button.destroy()
            backward_button.grid_forget()
            backward_button.destroy()
            arch_label.grid_forget()
            arch_label.destroy()
            layer_label.grid_forget()
            layer_label.destroy()
            label_layers.grid_forget()
            label_layers.destroy()
            save_button.grid_forget()
            save_button.destroy()
            canvas_frame.place_forget()
            
            
            # reconfig_button.grid(row=11, column=6, columnspan=6, padx=(100, 0), pady=20)
            
        def reconfigure():
                pass
            
    
        

        
        # # Load forward and backward arrow images
        # forward_path="C:/Users/Abhishek Srivastava/Downloads/forward_icon.png"
        # backward_path="C:/Users/Abhishek Srivastava/Downloads/Backward_icon.png"
    
        # # Resizing images of icons
        # def resize_image(image_path, new_width, new_height):
        #     original_image = Image.open(image_path)
        #     resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
        #     return ImageTk.PhotoImage(resized_image)
    
    
        # # Set the desired width and height for the resized image
        # new_width = 20
        # new_height = 15
    
    
        # # Resize the image
        # resized_forward_image = resize_image(forward_path, new_width, new_height)
        # resized_backward_image = resize_image(backward_path, new_width, new_height)
        # # image=resized_forward_image
        
        
        
        forward_button = Button(root1, text="Forward", command=next_page)
        backward_button = Button(root1, text="Backward", command=previous_page)
    
        save_button = Button(root1, text="Save and Create Layers", command=lambda: save_configuration(arch_label,layer_label,label_layers))
        global save_label
        save_label = Label(root1)
        save_label.grid_forget()
        
        reconfig_button= Button(root1,text="Reconfigure",command=reconfigure)
        reconfig_button.grid_forget()
    
    
        forward_button.grid(row=10, column=6, columnspan=6, padx=(460, 0), pady=10)
        backward_button.grid(row=10, column=2, columnspan=6, padx=(70, 0), pady=10)
    
        save_button.grid(row=11, column=6, columnspan=6, padx=(100, 0), pady=20)
        
    button_layer_details = Button(root1, text="Enter Layer Details", command=layer_details)
    button_layer_details.grid(row=1, column=6, columnspan=6, padx=(450, 0), pady=10)
    
    
root = Tk()
root.title("A GUI based application")

Label(root, text="Enter your Project Name:").grid(row=0, column=0, padx=10, pady=5)
entry_name = Entry(root, width=70)
entry_name.grid(row=1, column=1, padx=10, pady=5)

def on_enter(event):
    next_button.invoke()  # Invoke the next_button's command when Enter is pressed

next_button=Button(root, text="Next", command=next_window)
next_button.grid(row=2, column=1, padx=10, pady=5)
root.bind('<Return>', on_enter)

# Run the Tkinter event loop
root.mainloop()

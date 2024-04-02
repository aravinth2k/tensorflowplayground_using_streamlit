import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense,InputLayer
from keras.optimizers import SGD
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import pandas as pd
import numpy as np

filename = {'U Shape':'1.ushape.csv','Concentric Circle 1':'2.concerticcir1.csv',
            'Concentric Circle 2':'3.concertriccir2.csv','Linear Seperable':'4.linearsep.csv',
            'Outlier':'5.outlier.csv','Overlap':'6.overlap.csv','Xor':'7.xor.csv',
            'Two Spiral':'8.twospirals.csv','Random':'9.random.csv'}

with st.sidebar:
    dataset = st.selectbox('Select Dataset',('U Shape','Concentric Circle 1','Concentric Circle 2',
                                                    'Linear Seperable','Outlier','Overlap','Xor','Two Spiral','Random'))
    df = pd.read_csv(f'Data/{filename[dataset]}',header=None)
    fv = df.iloc[:, :2]
    cv = df.iloc[:, -1]
    fig_data, ax = plt.subplots(figsize=(4,4), constrained_layout=True)
    sns.scatterplot(x=fv.iloc[:, 0], y=fv.iloc[:, 1], hue=cv, ax=ax, legend=None)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])
    st.pyplot(fig_data)
        

with st.sidebar:
    st.header('Train Data using Neural Network')
    testsize = st.slider('Test data Size',min_value=20, max_value=90)
    random = st.number_input("Random Generator Seed",min_value=2,step=1)
    learning_rate = st.number_input("Learning Rate",value=None,format='%.3f')
    epoch = st.number_input("Epoch",min_value=10,step=1)
    no_of_hl = st.number_input("Number of Hidden layer",min_value=1,step=1)
    neurons = []
    activation = []
    for i in range(no_of_hl):
        neurons.append(st.number_input(f"Number of Neuron for {i+1} layer",min_value=1,step=1))
        activation.append(st.selectbox(f'Select Activation Function for {i+1} layer',('sigmoid','tanh','relu','softmax','linear')))
    train = st.button('Train',type='primary')

if train:
    std=StandardScaler()
    
    model = Sequential()
    sgd = SGD(learning_rate=learning_rate)
    model.add(InputLayer(shape=(2,)))

    for i in range(no_of_hl):
        model.add(Dense(neurons[i], activation=activation[i], use_bias=True))
    X_train, X_test, y_train, y_test = train_test_split(fv, cv, test_size=(testsize/100), random_state=42)
    
    model.add(Dense(1, activation='sigmoid', use_bias=True))
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    
    x_train = std.fit_transform(X_train)
    x_test = std.transform(X_test)
    with st.status('Training...'):
        st.write('Training Started')
        history = model.fit(x_train, y_train, epochs=epoch, validation_split=(testsize/100))
        st.write('Model Trained Successfully')
        
        fig_losses, ax_l = plt.subplots(figsize=(5,5), constrained_layout=True)
        sns.lineplot(x=range(1,epoch + 1),y=history.history["loss"],label="train_loss", ax=ax_l)
        sns.lineplot(x=range(1,epoch + 1),y=history.history["val_loss"],label="val_loss", ax=ax_l)
        plt.legend()
        plt.show()
    st.pyplot(fig=fig_losses)

    with st.status('Ploting Decison Boundary'):
        fig_db, ax1 = plt.subplots(figsize=(8,8), constrained_layout=True)
        st.write('Please Wait')
        plot_decision_regions(x_test, y_test.values.astype(np.int_), clf=model, ax=ax1)
        st.write('Plot Created')
    st.pyplot(fig_db)
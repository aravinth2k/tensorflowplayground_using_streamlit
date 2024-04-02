import streamlit as st
from sklearn.datasets import make_classification,make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense,InputLayer
from keras.optimizers import SGD
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

with st.sidebar:
    st.header('Random Data Generator')
    problem_type = st.selectbox('Select Task Type',('Classification','Regression'))
    samplesize = st.number_input("Sample Dataset Size",min_value=200,step=10)
    random = st.number_input("Random Generator Seed",min_value=2,step=1)
    flag = st.button('Generate',type='primary')

    if problem_type == 'Classification':
        fv,cv = make_classification(n_samples=samplesize,n_features=2,n_informative=2,n_redundant=0,n_repeated=0,
                                  n_classes=2,class_sep=3,random_state=random)
        output_act_func = 'sigmoid'
    elif problem_type == 'Regression':
        fv,cv = make_regression(n_samples=samplesize, n_features=2,random_state=random)
        output_act_func = 'linear'
        

if flag:
    fig_data, ax = plt.subplots(figsize=(8,8), constrained_layout=True)
    if problem_type == 'Classification':
        sns.scatterplot(x=fv[:, 0], y=fv[:, 1], hue=cv, ax=ax)
    elif problem_type == 'Regression':
        sns.scatterplot(x=fv[:, 0], y=fv[:, 1], ax=ax)
    st.pyplot(fig_data)

with st.sidebar:
    st.header('Train Data using Neural Network')
    testsize = st.slider('Test data Size',min_value=20, max_value=90)
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
    fig_data, ax = plt.subplots(figsize=(8,8), constrained_layout=True)

    std=StandardScaler()
    
    model = Sequential()
    sgd = SGD(learning_rate=learning_rate)
    model.add(InputLayer(shape=(2,)))

    for i in range(no_of_hl):
        model.add(Dense(neurons[i], activation=activation[i], use_bias=True))
    if problem_type == 'Classification':
        X_train, X_test, y_train, y_test = train_test_split(fv, cv, test_size=(testsize/100), stratify=cv, random_state=random)
        sns.scatterplot(x=fv[:, 0], y=fv[:, 1], hue=cv, ax=ax)
        model.add(Dense(1, activation=output_act_func, use_bias=True))
        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    elif problem_type == 'Regression':
        X_train, X_test, y_train, y_test = train_test_split(fv, cv, test_size=(testsize/100), random_state=random)
        sns.scatterplot(x=fv[:, 0], y=fv[:, 1], ax=ax)
        model.add(Dense(1, activation=output_act_func, use_bias=True))
        model.compile(optimizer=sgd, loss='mean_squared_error', metrics=["mse"])
    st.pyplot(fig_data)
    x_train = std.fit_transform(X_train)
    x_test = std.transform(X_test)
    with st.status('Training...'):
        st.write('Training Started')
        history = model.fit(x_train,y_train,epochs=epoch,validation_split=(testsize/100))
        st.write('Model Trained Successfully')
        
        fig_losses, ax_l = plt.subplots(figsize=(5,5), constrained_layout=True)
        sns.lineplot(x=range(1,epoch + 1),y=history.history["loss"],label="train_loss", ax=ax_l)
        sns.lineplot(x=range(1,epoch + 1),y=history.history["val_loss"],label="val_loss", ax=ax_l)
        plt.legend()
        plt.show()
        st.pyplot(fig=fig_losses)

    if problem_type == 'Classification':
        with st.status('Ploting Decison Boundary'):
            fig_db, ax1 = plt.subplots(figsize=(8,8), constrained_layout=True)
            st.write('Ploting Decision Boundary')
            st.write('Please Wait')
            plot_decision_regions(x_test,y_test,clf=model, ax=ax1)
            st.write('Plot Created')
        st.pyplot(fig_db)



    


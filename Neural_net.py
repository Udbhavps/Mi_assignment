
"""
Input layer has 6 neurons
1 hidden layer has 3 neurons and
the output layer has 1 neuron

weight matrices:
include one 6 x 3 for outerlayer and 3 x 1 for hidden layer.

Activation function:
Here the activation function used is sigmoid function for all neurons in the model.
And its derivative for backward propagation.

Loss function:
Cross entropy.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class NN:

	def CM(self,y_test,y_test_obs):
		'''
		Prints confusion matrix
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model
		'''

		for i in range(len(y_test_obs)):
			if(y_test_obs[i]>0.6):
				y_test_obs[i]=1
			else:
				y_test_obs[i]=0

		cm=[[0,0],[0,0]]
		fp=0
		fn=0
		tp=0
		tn=0

		for i in range(len(y_test)):
			if(y_test[i]==1 and y_test_obs[i]==1):
				tp=tp+1
			if(y_test[i]==0 and y_test_obs[i]==0):
				tn=tn+1
			if(y_test[i]==1 and y_test_obs[i]==0):
				fp=fp+1
			if(y_test[i]==0 and y_test_obs[i]==1):
				fn=fn+1
		cm[0][0]=tn
		cm[0][1]=fp
		cm[1][0]=fn
		cm[1][1]=tp

		p= tp/(tp+fp)
		r=tp/(tp+fn)
		f1=(2*p*r)/(p+r)

		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")

# Sigmoid activation Function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of sigmoid function:
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

if __name__ == '__main__':
    td = NN()

    # Importing the dataset
    data = pd.read_csv('processed_LBW_data.csv')
    '''
    #imputation of dataset
    to_drop = ['Residence',
               'Education',
               'Delivery phase']
    dataset.drop(to_drop, inplace=True, axis=1)
    dataset['Age'].fillna((dataset['Age'].mean()),inplace=True)
    dataset['Weight'].fillna((dataset['Weight'].mean()),inplace=True)
    dataset['HB'].fillna((dataset['HB'].mean()),inplace=True)
    dataset['BP'].fillna((dataset['BP'].mean()),inplace=True)
    #dataset['Delivery phase'].fillna((dataset['Delivery phase'].mode()[0]),inplace=True)
    #dataset['Residence'].fillna((dataset['Residence'].mode()[0]),inplace=True)
    #dataset['Education'].fillna((dataset['Education'].mode()[0]),inplace=True)
    '''

    #here three columns are eliminated as they have the same values throughout the column 
    #hence does not show any significant changes in training of the model
    x = data[['Community','Age','Weight','HB','IFA','BP']]
    y = data[['Result']]
    
    #using sklearn for the data split 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
    
    #generating the weight matrices
    hidden_layer = np.random.rand(6,3)
    output_layer = np.random.rand(3,1)
    learning_rate = 0.4
    
    # Converting to numpy
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()
   
    for reps in range(30000):

        # Forward Propogation with the sigmoid activation function for all neurons
        hidden_nn = np.dot(x_train, hidden_layer)
        hidden_out = sigmoid(hidden_nn)

        output_nn = np.dot(hidden_out, output_layer)
        output_out = sigmoid(output_nn)
        
        # Backpropogation of outer layer 

        loss_ou = output_out - y_train
        blo_bho = sigmoid_derivative(output_nn)
        bho_bbo = hidden_out
        loss_outer = np.dot(bho_bbo.T, loss_ou * blo_bho)
        
        #loss function cross entropy 
        error = (x_train * np.log(1e-15 + output_out))
        # Backpropogation of hidden layer 

        loss_hid = loss_ou * blo_bho
        bho_bhh = output_layer
        loss_bhh = np.dot(loss_hid , bho_bhh.T)
        bhh_bho = sigmoid_derivative(hidden_nn)
        bhh_bbh = x_train
        loss_hidden = np.dot(bhh_bbh.T, bhh_bho * loss_bhh)
        
        hidden_layer -= learning_rate * loss_hidden
        output_layer -= learning_rate * loss_outer
  

	# Testing the model on the test data 
    hidden = np.dot(x_train, hidden_layer)
    hidden_fn = sigmoid(hidden)

    output = np.dot(hidden_fn, output_layer)
    output_fn = sigmoid(output)

    td.CM(y_test,output_fn)
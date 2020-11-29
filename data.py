

import pandas as pd


if __name__ == '__main__':

    # Importing the dataset
    dataset = pd.read_csv('LBW_Dataset.csv')
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
    
    dataset.to_csv('raw_data.csv', index=False)
    #dataset.to_excel('raw_data.xls',index=False)
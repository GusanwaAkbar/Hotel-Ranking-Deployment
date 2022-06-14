from django.apps import AppConfig
import html
import pathlib
from pathlib import Path
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers



class FastmodelConfig(AppConfig):
    
    MODEL_PATH = Path("model")
    
    MODEL = MODEL_PATH/'model_weight2.h5'
    
    #default_auto_field = 'django.db.models.BigAutoField'
    DATA = MODEL_PATH/'hotels3.csv'


    df = pd.read_csv(DATA)


    name = 'fastmodel'


    def return_train_features(df=df):

        facilities_columns = ['Food and Drinks','Hotel Services','In-room Facilities', 'Business Facilities', 'Nearby Facilities', 'Public Facilities', 'General', 'Things to Do', 'Accessibilty', 'Connectivity', 'Transportation', 'Kids and Pets', 'Sports and Recreations', 'Shuttle Service']

        facilities_columns.reverse()

        for index , row in df.iterrows():
            
            # split per fasil and akomod
            arr = row['Facil + Akomod'].splitlines() 
            #iterate over fasil and akomod

            i = 0
            count = 0

            for  item in reversed(arr):
                count += 1
                if item in facilities_columns:
                    df.at[index,item ] = count
                    count = 0
                    i += 1
                
        df = df.fillna(0)

        for index , row in df.iterrows():
            
            # split per fasil and akomod
            arr = row['Places Nearby'].splitlines() 
            
            #iterate over fasil and akomod

            i = 0
            count = 0
            
            for ind, item in enumerate(arr):
                itemsplits = item.split()
                for x in itemsplits:
                    if x.isdigit():
                        if itemsplits[1] == "km":
                            meters = itemsplits[0] * 1000
                        else:
                            meters = itemsplits[0]
                            
                        if meters.isdigit():
                            df.at[index,arr[ind-1]] = meters
                            #print(meters)
                            
        df = df.fillna(15000)

        df.Harga = df['Harga'].str.replace('.','', regex = True)
        df.Harga = df['Harga'].str.replace(',','.', regex = True)
        df.Harga = df['Harga'].astype(float).astype(int)

        df.Reviews = df['Reviews'].str.replace('.','', regex = True)
        df.Reviews = df['Reviews'].str.replace(',','.', regex = True)
        df.Reviews = df['Reviews'].astype(float).astype(int)

        c = df.select_dtypes(object).columns
        df[c] = df[c].apply(pd.to_numeric,errors='coerce')

        a = []

        for x in range(90,1,-1):
            a.append(x)
            
        print(a)

        df['Score'] = a

        train_features = df.drop(['Hotel','Score','Places Nearby','Facil + Akomod','Fast Food', 'Shop & Gifts', 'Business',
            'Transportation Hub', 'Casual Dining', 'Nightlife', 'Park & Zoo',
            'Public Service', 'Arts & Sciences', 'Fine Dining', 'Sport',
            'Quick Bites', 'Education', 'Street Food', 'Activity & Games', 'Cafe',
            'Entertainment', 'Food Court', 'Sight & Landmark'], axis = 1)

        train_labels = df['Score']


        return train_features


    def preprocessing_output():

        data = pd.read_csv('https://storage.googleapis.com/data-hotel/list-hotel/Data%20API.csv')

        return data



    def large_model(train_features=return_train_features()):

        regularizer = 0.000001
        dropout = 0
        schedul = -0.0001
        lr = 0.001
    

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))
        print(normalizer.mean.numpy())


        model = keras.Sequential([
        normalizer,
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(regularizer)),
            layers.Dropout(dropout),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(regularizer)),
            layers.Dropout(dropout),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(regularizer)),
            layers.Dropout(dropout),
        layers.Dense(1)
        ])
        
        return model



    def return_predict(train_features = return_train_features()  , model=large_model(   return_train_features() ), weights = MODEL  ):

        dnn = model
        
        dnn.load_weights(weights)

        predict = dnn.predict(train_features)


        return predict




    score = return_predict()

    data = preprocessing_output()

    data['Score'] = score

    sorted_data = data.sort_values(by=['Score'])

    final_data = sorted_data.to_json(orient="index")

    print(final_data)
    print('lonte')


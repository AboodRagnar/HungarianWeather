import keras
from keras.layers import Activation,LSTM,Dropout,CuDNNLSTM
from keras.models import Sequential
from  keras.layers.core import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
ForTemp=MinMaxScaler(feature_range=(-1,1))
ToallButTemp=MinMaxScaler(feature_range=(0,1))

class ToProcessData:

    
    def __init__(self):
         self.GetData = []
         self.X=[]
         self.Y=[]
         self.Testx=[]
         self.Testy=[]
         self.MappingUtilities = {}
         
    
      


    def MappingObjectToBeEnumerated(self,map,Data):
      newEnu=[]
      for  i in Data:
        newEnu.append(map.index(i))  
      return newEnu        

    def GetAndCleanRawDataWeatherHistory(self,Path):
      
      #load and clean
      self.GetData=pd.read_csv(Path)
      self.GetData.drop(columns=['Formatted Date','Loud Cover','Summary'],inplace=True)
        
    
      
      Map1={k:v for v,k in enumerate(self.GetData['Daily Summary'].unique())}
      self.MappingUtilities.update({os.path.basename(Path)+'/Daily Summary':Map1})
      self.GetData['Daily Summary']=self.MappingObjectToBeEnumerated(list(Map1.keys()),self.GetData['Daily Summary'])
      Map1={k:v for v,k in enumerate(self.GetData['Precip Type'].unique())}
      
      self.MappingUtilities.update({os.path.basename(Path)+'/Precip Type':Map1})
      self.GetData['Precip Type']=self.MappingObjectToBeEnumerated(list(Map1.keys()),self.GetData['Precip Type'])

       #normalizing data
      self.GetData.iloc[:,1:3]=ForTemp.fit_transform(self.GetData.iloc[:,1:3])
      self.GetData.iloc[:,:1]=ToallButTemp.fit_transform(self.GetData.iloc[:,:1])
      self.GetData.iloc[:,4:]=ToallButTemp.fit_transform(self.GetData.iloc[:,4:])
      

      self.X=self.GetData.drop(columns=['Precip Type']).values
      self.Y=self.GetData['Precip Type'].values
      self.Testx=self.GetData.iloc[91809:,1:]
      self.Testx=self.Testx.values
      self.Testy=self.Y[91809:]

      

      

      


      #shaping in proper format
      TempX=[]
      TempY=[]
      TempTesx=[]
      TempTesY=[]
      for i in range(60,self.X.shape[0]-4645):
        TempX.append(self.X[i-60:i,:])
        TempY.append(self.Y[i])
       
      for i in range(60,self.Testx.shape[0]):
        TempTesx.append(self.Testx[i-60:i,:])
        TempTesY.append(self.Testy[i])
      
      
      self.X= np.array(TempX)
      
      self.Y=np.array(TempY)
      self.Testx= np.array(TempTesx)
      self.Testy=np.array(TempTesY)

      

      return 

    


    
    def GetKeyByValue(self,Value,MappingName):
    
     if MappingName in self.MappingUtilities.keys():
        Map=self.MappingUtilities.get(MappingName)
        
        for k,v in Map.items():

         if v==Value:
            
           return k

        return 'value does not exist'    
            
     

         
     else:      
        return "Mapping fliter does not exist"

    def SplitSequencesntoTimeStep(self,X,Y,n_step,normlized_len):

      Xseq=[]
      Yseq=[]
      
      if type(X).__module__!=pd.__name__ or  type(Y).__module__!=pd.__name__ : 
        if type(X).__module__!=np.__name__ or  type(Y).__module__!=np.__name__ :
           print('Warning entered data must be numpy or supporting numpy type')
           return np.array(Xseq),np.array(Yseq) 
        
      

      if np.ndim(X)==1 and np.ndim(Y)==1:
         for itr in range(normlized_len):
            if n_step+itr<=normlized_len:
              Xseq.append(X[itr:itr+n_step])
              Yseq.append(Y[n_step+itr-1]) 

      elif np.ndim(X)>1 and np.ndim(Y)==1:
        
          for itr in range(normlized_len):
            if n_step+itr<=normlized_len:
              Xseq.append(X[itr:itr+n_step,:])
              Yseq.append(Y[n_step+itr-1]) 

       

      elif np.ndim(X)>1 and np.ndim(Y)>1:
        
          for itr in range(normlized_len):
             if n_step+itr<=normlized_len:
              Xseq.append(X[itr:itr+n_step,:])
              Yseq.append(Y[n_step+itr-1,:]) 
        
        



      return np.array(Xseq),np.array(Yseq)  

      

print('hhh')
      

ProcessedData=ToProcessData()
# x=np.array([[4,69,2,4,685,68,38,0.33,-44,-0.36],[-4,169,0.2,4,685,-568,38,0.33,244,50.36],[304,169,200.2,54,-68,-568,38,0.33,244,10.86],[0,-9,700.2,4,685,-568,1038,2000.33,894,-2050.36]])
# y=np.array([[0,10],[-36,0.0005],[18,0],[-0.0086,200.0005]])
# print(x.shape)
# x1,y1=ProcessedData.SplitSequencesntoTimeStep(x,y,2,4)
# print(x1)
# print(y1)
#ProcessedData.GetAndCleanRawDataWeatherHistory('T3/Predict weather/weatherHistory.csv')


# print(ProcessedData.X.shape)
# Model=Sequential()


# Model.add(CuDNNLSTM(units=50,return_sequences=True,input_shape=(ProcessedData.X.shape[1],8)))
# Model.add(Dropout(0.2))

# Model.add(CuDNNLSTM(units=50,return_sequences=True))
# Model.add(Dropout(0.2))
# Model.add(CuDNNLSTM(units=50,return_sequences=True))
# Model.add(Dropout(0.2))

# Model.add(CuDNNLSTM(units=50))
# Model.add(Dropout(0.2))
# Model.add(Dense(units=3,activation='softmax'))

# Model.compile(Adam(0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# Model.fit(ProcessedData.X,ProcessedData.Y,epochs=50,batch_size=32)

# Model.save('HungarianWeather.h5')


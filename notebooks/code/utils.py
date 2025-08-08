import pandas as pd
import zipfile
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from CustomTransformers import TimeSeriesFreqRegularization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_dataset_folder():
    return '../../data/'


def get_mission_folder(mission_name):
    dataset_folder = get_dataset_folder()
    return dataset_folder + '/' + mission_name + '/'

def get_channels_file(mission_name):
    mission_folder = get_mission_folder(mission_name)
    return mission_folder + 'channels.csv'


def get_channels(mission_name):
    df = pd.read_csv(get_channels_file(mission_name))
    return df.to_dict(orient='records')

def load_channel(mission_name, channel):
    mission_folder = get_dataset_folder() + '/' + mission_name + '/'
    #channels_file = mission_folder + 'channels.csv'
    
    fpath = mission_folder + 'channels/' + channel['Channel'] + '.zip'
    with zipfile.ZipFile(fpath) as z:
        with z.open(channel['Channel']) as f:
            channel['data'] = pd.read_pickle(f)
            channel['data'].index = pd.to_datetime(channel['data'].index)
            channel['data'].sort_index(inplace=True)
            


    return channel

def get_channel(mission_name, channel_name):
    df = pd.read_csv(get_channels_file(mission_name))
    channel = df[df['Channel'] == channel_name].to_dict(orient='records')[0]
    return load_channel(mission_name, channel)


def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int, window: int, method: str, order: int) -> list:
    
    total_len = train_len + horizon
    end_idx = train_len
    
    if method == 'mean':
        pred_mean = []
        
        for i in range(train_len, total_len, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))
            
        return pred_mean

    elif method == 'last':
        pred_last_value = []
        
        for i in range(train_len, total_len, window):
            last_value = df[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))
            
        return pred_last_value
    elif method == 'MA':
        pred_MA = []
        
        for i in range(train_len, total_len, window):
            
            model = SARIMAX(df[:i], order=(0,0,order))
            #res es diferente a sklearn, res es el resultado y las herramientas para realizar predicciones
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_MA.extend(oos_pred)
            
        return pred_MA
    
    elif method == 'AR':
        pred_AR = []
        
        for i in range(train_len, total_len, window):
            model = SARIMAX(df[:i], order=(2,0,0))
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_AR.extend(oos_pred)
            
        return pred_AR
    

class SingleTlmyManager:
    def __init__(self, mission_name, channel_name, start_date=None, end_date=None):
        self.mission_name = mission_name
        self.channel_name = channel_name

        df                  = pd.read_csv(get_channels_file(mission_name))
        channel             = df[df['Channel'] == channel_name].to_dict(orient='records')[0]


        mission_folder = get_dataset_folder() + '/' + mission_name + '/'
        #channels_file = mission_folder + 'channels.csv'
        fpath = mission_folder + 'channels/' + self.channel_name + '.zip'
        with zipfile.ZipFile(fpath) as z:
            with z.open(self.channel_name) as f:
                self.data = pd.read_pickle(f)
                self.data.index = pd.to_datetime(self.data.index, utc=True)
                self.data.sort_index(inplace=True)
                self.data.index = self.data.index.tz_convert('UTC')
                if start_date is not None and end_date is not None:
                    self.data = self.data[(self.data.index >= start_date) & (self.data.index <= end_date)]
        

        self.subsystem      = channel['Subsystem']
        self.channel        = channel
        self.PhysicalUnit   = channel['Physical Unit']
        self.group          = channel['Group']
        self.target         = channel['Target']
        self.labels         = False
        self.freq_          = None 
    
        

    def get_channel(self):
        return self.channel

    def add_labels(self):
        df = self.data
        df['isAnomaly'] = False
        df['anomalyId'] = None
        df_labels = pd.read_csv(get_dataset_folder() + '/' + self.mission_name + '/labels.csv')
        df_labels['StartTime']  = pd.to_datetime(df_labels['StartTime'], utc=True)
        df_labels['EndTime']    = pd.to_datetime(df_labels['EndTime'], utc=True)
        df_labels = df_labels.loc[df_labels['Channel'] == self.channel_name]
        for start, end, ID in zip(df_labels['StartTime'], df_labels['EndTime'], df_labels['ID']):
            mask = (df.index >= start) & (df.index <= end)
            df.loc[mask, 'isAnomaly'] = True
            df.loc[mask, 'anomalyId'] = ID
        self.data   = df
        self.labels = True
        return self.data

    def regularize_freq(self):
        transformer     = TimeSeriesFreqRegularization(columns=self.channel_name)
        self.data       = transformer.fit_transform(self.data)
        self.freq_      = self.data.index.freq
        self.freqstr_   = self.data.index.freqstr
        return self.data
    
    def get_anomaly_groups(self):
        if not self.labels:
            raise ValueError("Labels not added. Please call add_labels() first.")
        df_anomalies = self.data.groupby("anomalyId").agg(
            start_index=("channel_44", lambda x: x.index.min()),
            end_index=("channel_44", lambda x: x.index.max()),
            count=("channel_44", "count")
        )
        df_anomalies = df_anomalies.sort_values(by="start_index")
        df_anomalies = df_anomalies.reset_index()
        
        return df_anomalies[df_anomalies.anomalyId != np.nan]
    



def split_and_scale_3d(X, y, test_size=0.2, shuffle=False, scaler_cls=StandardScaler):
    """
    Divide y escala un array 3D para modelos como LSTM sin fuga de datos.
    
    Parámetros
    ----------
    X : np.ndarray
        Array de entrada con forma (n_muestras, time_steps, n_features)
    y : np.ndarray
        Etiquetas, 1D o 2D
    test_size : float
        Proporción de test (default=0.2)
    shuffle : bool
        Si True mezcla las muestras (para series de tiempo usar False)
    scaler_cls : class
        Clase de escalador de sklearn (ej. StandardScaler o MinMaxScaler)
    
    Retorna
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    # 1️⃣ División
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle
    )
    
    # 2️⃣ Ajuste del escalador solo con entrenamiento
    n_train, time_steps, n_features = X_train.shape
    scaler = scaler_cls()
    X_train_2d = X_train.reshape(-1, n_features)
    scaler.fit(X_train_2d)
    
    # 3️⃣ Transformación manteniendo forma original
    X_train_scaled = scaler.transform(X_train_2d).reshape(n_train, time_steps, n_features)
    
    n_test = X_test.shape[0]
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(n_test, time_steps, n_features)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataHandler:
    def __init__(self, file_path, test_size=0.3, random_state=42):
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def load_data(self):
        df = pd.read_excel(self.file_path, engine='openpyxl')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.reshape(-1, 1)
        return X, y

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def scale_data(self, X_train, X_test, y_train):
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        return X_train_scaled, X_test_scaled, y_train_scaled

    def inverse_transform(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled)
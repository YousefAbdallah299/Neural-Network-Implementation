import numpy as np
from NeuralNetwork import NeuralNetwork
from DataHandler import DataHandler

def main():
    data_handler = DataHandler('concrete_data.xlsx')

    X, y = data_handler.load_data()
    X_train, X_test, y_train, y_test = data_handler.split_data(X, y)
    X_train_scaled, X_test_scaled, y_train_scaled = data_handler.scale_data(X_train, X_test, y_train)


    nn = NeuralNetwork(input_size=4, hidden_size=16, output_size=1, learning_rate=0.01, epochs=1000)
    nn.train(X_train_scaled, y_train_scaled, acceptable_error=20, scaler_y=data_handler.scaler_y)


    predictions_scaled = nn.predict(X_test_scaled)
    predictions = data_handler.inverse_transform(predictions_scaled)



    print("\n\nTests:\n")
    for i in range(len(predictions)):
        print(f"Predicted: {predictions[i][0]:.2f}, Actual: {y_test[i][0]:.2f}")

    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    print(f"\nMean absolute percentage error: {mape:.2f}%")

if __name__ == "__main__":
    main()
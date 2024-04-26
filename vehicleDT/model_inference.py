from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_test_pred(df):
    # Load the autoencoder model
    model = load_model("autoencoder.keras")

    # Read the CSV file
    df = pd.read_csv(data_path)

    # Extract relevant columns
    rpm_speed_test = df[['time', 'Engine RPM (rpm)', 'Speed (GPS) (km/h)']]
    # print(rpm_speed_test['time'])

    # Fill missing values
    rpm_speed_test['Engine RPM (rpm)'] = rpm_speed_test['Engine RPM (rpm)'].fillna(method='ffill')
    rpm_speed_test['Engine RPM (rpm)'] = rpm_speed_test['Engine RPM (rpm)'].fillna(method='bfill')
    rpm_speed_test = rpm_speed_test.dropna(subset=['Speed (GPS) (km/h)'])

    # Scale the data
    scaler = StandardScaler()
    scalar = scaler.fit(rpm_speed_test[['Speed (GPS) (km/h)']])
    rpm_speed_test['Speed (GPS) (km/h)'] = scalar.transform(rpm_speed_test[['Speed (GPS) (km/h)']])

    scaler = StandardScaler()
    scalar = scaler.fit(rpm_speed_test[['Engine RPM (rpm)']])
    rpm_speed_test['Engine RPM (rpm)'] = scalar.transform(rpm_speed_test[['Engine RPM (rpm)']])

    # Create sequences
    def create_sequences(values, time_steps=1):
        output = []
        for i in range(len(values) - time_steps + 1):
            output.append(values[i: (i + time_steps)])
        return np.array(output)

    testX = create_sequences(rpm_speed_test.values, 288)
    testX = np.reshape(testX, (testX.shape[0], 288, 3))

    timeX = create_sequences(rpm_speed_test[['time']].values, 288)
    timeX = np.reshape(timeX, (timeX.shape[0], 288, 1))

    # Predict
    test_pred = model.predict(testX[:, :, 1:].astype(np.float32))

    # Error treshold
    test_mae_loss = np.mean(np.abs(test_pred - testX[:, :, 1:]), axis=1)

    # Add time column back
    # timeX = timeX[:, timeX.shape[1] // 2, 0]
    timeX = timeX[:, 0, 0]
    test_mae_loss = np.concatenate((timeX[:, np.newaxis], test_mae_loss), axis=1)


    return test_mae_loss



# Usage
data_path = r"local_data\2024-03-11_17-58-05.csv"
test_mae_loss = get_test_pred(data_path)
print(test_mae_loss.shape)
# print(test_mae_loss)

print(test_mae_loss[3])
print(test_mae_loss[7])
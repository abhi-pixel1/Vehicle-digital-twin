# import pickle
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def standardize(df):
    scaler = StandardScaler()
    scalar = scaler.fit(df)
    std_df = scalar.transform(df)
    return std_df



def fillNan(df):
    df['Engine RPM (rpm)'] = df['Engine RPM (rpm)'].ffill()
    df['Engine RPM (rpm)'] = df['Engine RPM (rpm)'].bfill()

    df = df.dropna(subset = ['Speed (GPS) (km/h)'])

    return df


def create_sequences(values, time_steps=1):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.array(output)



def inp_prep(path):
    df = pd.read_csv(path)
    # df['time'] = pd.to_datetime(df['time'])
    df = df[['time', 'Engine RPM (rpm)', 'Speed (GPS) (km/h)']]


    rpm_speed = fillNan(df)

    rpm_speed.loc[:, 'Engine RPM (rpm)'] = standardize(rpm_speed[['Engine RPM (rpm)']])
    rpm_speed.loc[:, 'Speed (GPS) (km/h)'] = standardize(rpm_speed[['Speed (GPS) (km/h)']])


    rpm_speed = create_sequences(rpm_speed[['Speed (GPS) (km/h)', 'Engine RPM (rpm)']].values, 288)
    rpm_speed = np.reshape(rpm_speed, (rpm_speed.shape[0], 288, 2))

    # rpm_speed = rpm_speed.reshape(1, 288, 2)
    return rpm_speed






def guess(inp):
    model = load_model("autoencoder.keras")

    # model = pickle.load(open('autoencoder.pkl','rb'))

    pred = model.predict(inp)
    return pred


def compare_treshold(output, input):
    test_mae_loss = np.mean(np.abs(output - input), axis=1)

    threshold_speed = np.max(test_mae_loss[:][:,0])
    threshold_rpm = np.max(test_mae_loss[:][:,1])

    return (threshold_speed, threshold_rpm)


def get_5_val(df, index, col):
    df = df[col].dropna()
    df = df.iloc[index:index+5].tolist()
    return df



def last_10(index, df, col_name):


    # Get the index of the last 5 non-NaN elements before the given index
    non_nan_indices = df[col_name].dropna().index
    non_nan_indices_before_index = non_nan_indices[non_nan_indices < index][-10:]
    
    # Retrieve the corresponding values
    last_5_non_nan_values = [df.at[i, col_name] for i in non_nan_indices_before_index]
    last_5_non_nan_time = [df.at[i, 'time'] for i in non_nan_indices_before_index]
    
    data = [last_5_non_nan_time, last_5_non_nan_values]
    return data



def get_test_pred(df):
    # Load the autoencoder model
    model = load_model("autoencoder.keras")

    # Extract relevant columns
    rpm_speed_test = df[['time', 'Engine RPM (rpm)', 'Speed (GPS) (km/h)']]

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




# Example usage:
# Suppose df is your DataFrame and 'col' is the column name
# and idx is the index you're interested in.
# last_5_non_nan_values = last_5_non_nan_before_index(idx, df, 'col')




# input = inp_prep(r"D:\AA_WORKSPACE\dtxr\dtxr_mini\local_data\2024-03-11_17-58-05.csv")
# # # input = input.reshape(1, 288, 2)
# pred = guess(input[0].reshape(1, 288, 2))
# print(pred)
# print(compare_treshold(pred, input))




# input = pd.read_csv(r"D:\dtxr\dtxr_mini\local_data\2024-03-16_19-38-51.csv")
# print(last_10(20, input, 'Engine RPM (rpm)'))

# df = pd.read_csv(r"D:\dtxr\dtxr_mini\local_data\2024-03-16_19-38-51.csv")
# latlog = df[['Latitude', 'Longtitude']]
# latlog = latlog.values.tolist()
# print(latlog[:100])
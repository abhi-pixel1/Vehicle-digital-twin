from flask import Flask, request, render_template, make_response, url_for, session
from flask_socketio import join_room, leave_room, send, SocketIO
from methods import last_10, get_test_pred
import time
from random import random
import json
import pandas as pd
import numpy as np
import math


thread_handler = 1

def pred_at_time_t(array, string):
    for row in array:
        if row[0] == string:
            return row
    return None

def is_anomalous(loss_val, threshold):
    if loss_val > threshold:
      return 1
    else:
      return 0

def angle_between_points(a, b, c):
    # Calculate vectors AB and BC
    AB = [b[0] - a[0], b[1] - a[1]]
    BC = [c[0] - b[0], c[1] - b[1]]

    # Calculate dot product of AB and BC
    dot_product = AB[0] * BC[0] + AB[1] * BC[1]

    # Calculate magnitudes of AB and BC
    mag_AB = math.sqrt(AB[0]**2 + AB[1]**2)
    mag_BC = math.sqrt(BC[0]**2 + BC[1]**2)

    # Calculate cosine of the angle between AB and BC
    cos_theta = dot_product / (mag_AB * mag_BC)

    # Calculate the angle in radians
    angle_rad = math.acos(cos_theta)

    # Convert angle to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg



def background_thread(index):
    global df
    global test_mae_loss
    # latlog = df[['Latitude', 'Longtitude']]
    # latlog = latlog.values.tolist()
    speed_err = 0
    rpm_err = 0


    for index, row in df.iloc[index+1:].iterrows():
        if thread_handler == 0:
            break
        
        rpm = row['Engine RPM (rpm)']
        speed = row['Speed (GPS) (km/h)']
        baro = row['Barometric pressure (kPa)']
        cool = row['Engine coolant temperature']
        cat = row['Catalyst temperature Bank 1 Sensor 1']
        latlong = row[['Latitude', 'Longtitude']].tolist()        
        

        data = {'time': row['time'], 'latlong': latlong}

        pred = pred_at_time_t(test_mae_loss, row['time'])

        if not np.isnan(rpm):
            data["rpm"] = rpm
        if not np.isnan(speed):
            data["speed"] = speed
        if not np.isnan(baro):
            data["baro"] = baro
        if not np.isnan(cool):
            data["cool"] = cool
        if not np.isnan(cat):
            data["cat"] = cat
        if pred is not None:
            # data['pred'] = pred
            speed_err = is_anomalous(pred[1], 0.3534215753463501)
            rpm_err = is_anomalous(pred[2], 0.2184191426907788)

            print(data, pred)

        data['speed_err'] = speed_err
        data['rpm_err'] = rpm_err



        # data = {'time': row['time'], 'rpm': rpm, 'speed': speed}
        socketio.emit("message", data)            
        socketio.sleep(1)



app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/', methods=['GET', 'POST'])
def hello():
    global df
    global test_mae_loss


    df = pd.read_csv(r"local_data\2024-03-16_18-35-20.csv")
    test_mae_loss = get_test_pred(df)


    test_mae_loss[3][1] = 1.0
    test_mae_loss[7][1] = 1.0


    rows = df.shape[0]
    
    start_time = df['time'][0]
    end_time = df.iloc[-1]['time']

    start_coords = df[['Latitude', 'Longtitude']].iloc[0].values.tolist()

    return render_template('index.html', rows=rows, start_time=start_time, end_time=end_time, start_coords=start_coords)


@socketio.on("connect")
def connect(auth):
    print('Client connected')


@socketio.on("play")
def play(data):
    global thread_handler
    thread_handler = 1
    print('play event')
    socketio.start_background_task(background_thread, int(data['index']))
    print(data['play'])


@socketio.on("pause")
def pause(data):
    global thread_handler
    thread_handler = 0
    print('pause event')
    print(data['pause'])


@socketio.on("sliderSync")
def sliderSync(data):
    global df
    global thread_handler
    thread_handler = 0


    last_10_speed = last_10(int(data['index']), df, 'Speed (GPS) (km/h)')
    last_10_rpm = last_10(int(data['index']), df, 'Engine RPM (rpm)')
    last_10_baro = last_10(int(data['index']), df, 'Barometric pressure (kPa)')
    last_10_cool = last_10(int(data['index']), df, 'Engine coolant temperature')
    last_10_cat = last_10(int(data['index']), df, 'Catalyst temperature Bank 1 Sensor 1')

    latlong = df[['Latitude', 'Longtitude']].values.tolist()[:int(data['index'])]

    data = {'time': df.iloc[int(data['index'])]['time'],
             'latlong': latlong, 
             'speed': last_10_speed, 
             'rpm': last_10_rpm, 
             'baro': last_10_baro, 
             'cool': last_10_cool,
             'cat': last_10_cat
             }
    socketio.emit("sliderSync", data)



@socketio.on("disconnect")
def disconnect():
    global thread_handler
    thread_handler = 0
    print("disconnected")


if __name__ == "__main__":
    socketio.run(app, debug=True, port=3030)
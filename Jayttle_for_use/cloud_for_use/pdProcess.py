import pandas as pd

def avr_pdProcess(file_path: str):
    avr_column = ['time','stationID', 'fixmode', 'yaw', 'tilt', 'range', 'pdop', 'sat_num']
    df = pd.read_csv(r"D:\Program Files (x86)\Software\OneDrive\PyPackages\avr_yesterday_data.txt", names=avr_column)

def met_pdProcess(file_path: str):
    met_column = ['time', 'stationID', 'temperature', 'humidness', 'pressure', 'windSpeed', 'windDirection']
    df = pd.read_csv(file_path, names=met_column)

def tiltmeter_pdProcess(file_path: str):
    tiltmeter_column = ['time', 'stationID', 'pitch', 'roll']
    df = pd.read_csv(file_path, names=tiltmeter_column)

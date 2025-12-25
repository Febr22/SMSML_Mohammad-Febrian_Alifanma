import requests
import time
import random

url = "http://localhost:5001/predict"

print("Mulai mengirim data dummy ke API...")

# Data dummy (sesuai dataset beras)
dummy_data = [
    {"Area": 15231, "Perimeter": 525.578979, "Major_Axis_Length": 229.749878, "Minor_Axis_Length": 85.093788, "Eccentricity": 0.928882},
    {"Area": 14656, "Perimeter": 494.311005, "Major_Axis_Length": 206.020065, "Minor_Axis_Length": 91.130127, "Eccentricity": 0.896919},
    {"Area": 13632, "Perimeter": 477.269989, "Major_Axis_Length": 199.135269, "Minor_Axis_Length": 87.730225, "Eccentricity": 0.897960}
]

while True:
    try:
        # Pilih data random
        payload = random.choice(dummy_data)
        
        # Kirim request
        response = requests.post(url, json=payload)
        
        print(f"Status: {response.status_code} | Respon: {response.text}")
        
        # Tunggu sebentar (random 0.5 - 2 detik)
        time.sleep(random.uniform(0.5, 2.0))
        
    except Exception as e:
        print(f"Error connecting: {e}")
        time.sleep(2)
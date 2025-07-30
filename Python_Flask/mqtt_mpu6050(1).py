import time
import json
from mpu6050 import mpu6050
import paho.mqtt.client as mqtt


sensor = mpu6050(0x68)

MQTT_BROKER = "192.168.50.9"
MQTT_PORT = 1883
MQTT_TOPIC = "mpu6050/accel"

client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

SEND_INTERVAL = 0.1

try:
    while True:
        accel = sensor.get_accel_data(g=True)
        gyro = sensor.get_gyro_data()

        payload = json.dumps({
            "accel": {
                "x": round(accel["x"], 3),
                "y": round(accel["y"], 3),
                "z": round(accel["z"], 3)
            },
            "gyro": {
                "x": round(gyro["x"], 3),
                "y": round(gyro["y"], 3),
                "z": round(gyro["z"], 3)
            },
            "timestamp": time.time()
        })

        client.publish(MQTT_TOPIC, payload)
        print(f"以傳送 {payload}")

        time.sleep(SEND_INTERVAL)

except KeyboardInterrupt:
    print("結束傳送")
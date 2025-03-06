import requests
import random
import time
import threading

# URL of the Flask API endpoint
url = 'http://127.0.0.1:5000/api/smart_band_data'

simulation_active = False
simulation_thread = None

def generate_vitals(user_id):
    """Generate random vital signs for a specific user."""
    heart_rate = round(random.uniform(60, 100), 2)
    systolic_bp = random.randint(100, 140)
    diastolic_bp = random.randint(60, 90)
    blood_pressure = f"{systolic_bp}/{diastolic_bp}"

    return {
        'user_id': user_id,
        'heart_rate': heart_rate,
        'blood_pressure': blood_pressure
    }

def simulate_data(user_id):
    """Simulate sending vitals data."""
    while simulation_active:
        data = generate_vitals(user_id)
        response = requests.post(url, json=data)

        if response.status_code == 201:
            print('Data sent successfully:', data)
        else:
            print('Failed to send data:', response.status_code, response.text)

        time.sleep(5)

def start_simulation(user_id):
    """Start the smart band simulation."""
    global simulation_active, simulation_thread
    if not simulation_active:
        simulation_active = True
        simulation_thread = threading.Thread(target=simulate_data, args=(user_id,), daemon=True)
        simulation_thread.start()
        print("Smart band simulation started.")

def stop_simulation():
    """Stop the smart band simulation."""
    global simulation_active
    if simulation_active:
        simulation_active = False
        if simulation_thread:
            simulation_thread.join()  # Wait for the simulation thread to finish
        print("Smart band simulation stopped.")

# Ensure this script is not executed standalone; it's meant to be run as part of the Flask app.
if __name__ == '__main__':
    print("This script is intended to be run within the Flask application.")

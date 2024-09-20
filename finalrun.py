import RPi.GPIO as GPIO
import time
import torch
import cv2
from torchvision import transforms, models
from PIL import Image
from picamera2 import Picamera2
import os

# Set GPIO mode
GPIO.setmode(GPIO.BCM)

# Define GPIO pins for Ultrasonic Sensor
TRIG = 14
ECHO = 18
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Define GPIO pin for Servo Motor
SERVO_PIN = 23  # Change this to your actual GPIO pin for the servo
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM frequency
servo.start(0)  # Initialize with a neutral duty cycle (0)

# Initialize the camera using picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (224, 224)}))  # Configured for still image capture
picam2.start()

# Define the transformation (same as in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize if required
])

# Load the trained model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('/home/Pranav_Srikarthik/Downloads/resnet18_model.pth', map_location=torch.device('cpu')))
model.eval()

def measure_distance():
    """Function to measure distance using ultrasonic sensor."""
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    
    start_time = time.time()
    stop_time = time.time()
    
    while GPIO.input(ECHO) == 0:
        start_time = time.time()
        
    while GPIO.input(ECHO) == 1:
        stop_time = time.time()
    
    elapsed_time = stop_time - start_time
    distance = (elapsed_time * 34300) / 2  # Speed of sound in cm/s
    return distance
def actuate_servo(direction):
    """Function to actuate the SG90 servo motor to 45 degrees."""
    print("Actuating servo...")

    if direction == 1:
        # Tilt left by 45 degrees
        print("Non-biodegradable")
        servo.ChangeDutyCycle(3.5)  # Move to 45 degrees
    elif direction ==0:
        # Tilt right by 45 degrees
        print("Biodegradale waste")
        servo.ChangeDutyCycle(6.5)  # Move to 45  degrees in the other direction

    time.sleep(1)  # Give time for the servo to reach the position
    servo.ChangeDutyCycle(5)  # Move servo back to the neutral position (0 degrees)
    time.sleep(1)
    servo.ChangeDutyCycle(0)  # Stop the servo (to avoid jitter)



try:
    while True:
        distance = measure_distance()
        print(f"Distance: {distance:.2f} cm")

        # If the object is within a certain range, analyze the image
        if distance < 10:  # Adjust threshold distance (e.g., 10 cm)
            print("Object detected! Analyzing image...")

            # Capture image as a JPEG file using picamera2
            image_path = '/home/Pranav_Srikarthik/captured_image.jpg'
            picam2.capture_file(image_path, format="jpeg")

            # Open the saved image using PIL for transformation
            pil_image = Image.open(image_path)

            # Debug: Save the PIL image to verify that it's in the correct format
            pil_image.save('/home/Pranav_Srikarthik/temp_preprocessed_image_corrected.jpg')

            # Transform image
            input_tensor = transform(pil_image).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Output prediction and confidence
                predicted_class = predicted.item()
                confidence_score = probabilities[0][predicted_class].item()
                print(f'Predicted class: {predicted_class} with confidence: {confidence_score:.2f}')
                if(predicted_class==0):
                    print("Biodegradable Object Detected")
                else:
                    print("Non-Biodegradable Object Detected")
                
                # Actuate the servo motor based on the prediction
                actuate_servo(predicted_class)
                
        time.sleep(3)  # Adjust as needed for performance

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    GPIO.cleanup()
    picam2.stop()
    servo.stop()  # Ensure the servo is stopped when the script exits

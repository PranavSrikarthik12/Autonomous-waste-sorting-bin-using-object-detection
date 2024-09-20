

# Autonomous Waste Sorting Bin using Raspberry Pi

This project is an autonomous waste sorting bin built using **Raspberry Pi**, **Pi Camera**, **Servo Motor**, **Ultrasonic Sensor**, and **Load Cell**. The system classifies waste into four categories: **Biodegradable**, **Non-Biodegradable**, **E-Waste**, and **Medical Waste** using a machine learning model trained on a dataset from Kaggle. The classified waste is then sorted into four compartments, and the weight of the waste is measured.

## Features
- **Automatic Waste Detection:** The ultrasonic sensor detects when waste is thrown into the bin.
- **Image Classification:** The Pi Camera captures the image of the waste, which is then sent to the Raspberry Pi for classification using a trained model.
- **Waste Sorting:** A servo motor actuates to direct the waste into one of four compartments based on the classification.
- **Weight Measurement:** A load cell measures the weight of the waste after sorting.
  
## Components Used
- **Raspberry Pi 4** (or similar)
- **Pi Camera** for image capture
- **Ultrasonic Sensor** to detect waste
- **Servo Motors** (MG996R, SG90) for waste sorting
- **Load Cell** for weight measurement
- **ESP32** for motor control

## Working Flow
1. **Waste Detection:** The ultrasonic sensor detects the presence of waste in the bin.
2. **Image Capture & Classification:** The Pi Camera captures an image, which is then classified using a machine learning model into one of four categories:
   - Biodegradable
   - Non-Biodegradable
   - E-Waste
   - Medical Waste
3. **Waste Sorting:** Based on the classification, the servo motor actuates and directs the waste into the appropriate compartment.
4. **Weight Measurement:** The load cell measures the weight of the waste, and the data is logged for further analysis.

## Dataset
The dataset used for training the waste classification model was obtained from Kaggle. The model has been trained to classify waste into four categories: Biodegradable, Non-Biodegradable, E-Waste, and Medical Waste.

## Usage
- Once the system is powered on and set up, throw waste into the bin.
- The system will automatically detect the waste, classify it, sort it, and measure its weight.

## Future Improvements
- Integration with cloud platforms like **Ubidots** for real-time data logging and monitoring.
- Improving classification accuracy by expanding the dataset.
- Adding more waste categories for better sorting.

---



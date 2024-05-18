# Gesture Volume Control Project

This project enables volume control for your computer through hand gestures. By leveraging hand tracking technology, it detects specific gestures and adjusts the volume accordingly. The system is designed to be modular, utilizing existing hand tracking modules for streamlined implementation.

### Packages Used:
- OpenCV
- MediaPipe
- Alsaaudio
- NumPy

### Three main functionalities:
1. **Hand Detection and Tracking:** Utilizes the MediaPipe library to detect and track the user's hand in real-time video streams.
2. **Gesture Recognition:** Analyzes hand gestures to recognize predefined movements, such as controlling volume based on the distance between fingers.
3. **Volume Adjustment:** Interprets recognized gestures to adjust the system's audio volume accordingly. It also includes features like toggling mute status.

### Usage
To use the Gesture Volume Control Project:
1. Ensure your computer has a webcam connected and accessible.
2. Run the provided Python script, which captures video from the webcam and processes it in real-time.
3. Perform hand gestures within the camera's view to control the volume:
   - **Adjusting Volume:** Move your thumb and index finger closer or farther apart to adjust the volume level.
   - **Setting Volume:** When the small finger is raised, adjust the volume to the desired level. Once the small finger is lowered for a brief moment         (e.g., a second), the volume is set to the adjusted level.

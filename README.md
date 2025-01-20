# Velora Secure Authentication

## Description
Velora Secure Authentication is a robust face recognition-based authentication system built with Python. It provides a secure and user-friendly way to register and authenticate users using facial biometrics. The system utilizes computer vision and machine learning techniques to ensure accurate face detection and matching, with additional liveness detection through eye blink verification.

## Features
- Face-based user registration
- Real-time face detection and recognition
- Secure storage of face encodings in MongoDB
- Interactive command-line interface
- Real-time video feedback during registration and authentication
- Robust error handling and system diagnostics
- Eye blink detection for liveness verification (Coming Soon)

## Prerequisites
- Python 3.8+
- OpenCV
- MongoDB
- Webcam/Camera device

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd velora-secure-authentication
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install opencv-python
pip install face-recognition
pip install pymongo
pip install numpy
```

4. Ensure MongoDB is installed and running:
```bash
sudo systemctl start mongod
sudo systemctl status mongod
```

## System Components

### face_detector.py
Core component that handles:
- Face detection and encoding
- Database interactions
- Camera initialization
- Image preprocessing
- Eye blink detection and liveness verification (Upcoming)

### register_user.py
Handles the user registration process:
- User data input validation
- Face capture and encoding
- Database storage
- Interactive user feedback
- Liveness verification through eye blink detection (Upcoming)

### identify_user.py
Manages the authentication process:
- Real-time face detection
- Face matching with stored encodings
- Visual feedback with bounding boxes
- Performance monitoring
- Eye blink verification for anti-spoofing (Upcoming)

## Usage

### User Registration
```bash
python register_user.py
```
Follow the prompts to:
1. Enter the user's name
2. Position face in front of camera
3. Complete eye blink verification (Coming Soon)
4. Confirm successful registration

### User Authentication
```bash
python identify_user.py
```
The system will:
1. Initialize camera
2. Detect faces in real-time
3. Match detected faces with registered users
4. Verify liveness through eye blink detection (Coming Soon)
5. Display results on screen

## Security Features

### Current Features
- Face encodings are stored securely in MongoDB
- Real-time validation
- Timeout and retry mechanisms
- Input validation and sanitization

### Upcoming Security Enhancements
- Liveness detection through eye blink verification
- Anti-spoofing measures
- Enhanced authentication confidence scoring

## Troubleshooting

### Camera Access Issues
1. Check camera permissions:
```bash
ls -l /dev/video*
```

2. Add user to video group:
```bash
sudo usermod -a -G video $USER
```

3. Install video utilities:
```bash
sudo apt-get install v4l-utils
```

### Database Issues
1. Verify MongoDB service is running:
```bash
sudo systemctl status mongod
```

2. Check MongoDB logs:
```bash
sudo tail -f /var/log/mongodb/mongod.log
```

## Performance Optimization
- Uses HOG model for efficient face detection
- Implements frame rate monitoring
- Includes camera warm-up period
- Optimized image preprocessing
- Efficient eye blink detection algorithms (Upcoming)

## Development Roadmap
1. Current Version
   - Basic face detection and recognition
   - User registration and authentication
   - Database integration

2. Next Release
   - Eye blink detection implementation
   - Liveness verification
   - Enhanced anti-spoofing measures

## Contributing
Contributions are welcome! Please read our contributing guidelines and code of conduct before submitting pull requests.

## License
MIT License
Copyright (c) 2024 Velora Secure Authentication
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contact


## Acknowledgments
- Face Recognition library
- OpenCV community
- MongoDB team
- All contributors and testers

# E-Waste Management Using Deep Learning

An intelligent robotic arm system that uses computer vision and machine learning to detect, classify, and sort electronic waste items using Particle Swarm Optimization (PSO) for path planning.

> **Note**: This repository contains a comprehensive implementation of an e-waste management system using YOLOv8 and robotics integration.

## 🤖 Features

- **Real-time Object Detection**: Uses YOLO (You Only Look Once) for detecting electronic waste items
- **Toxicity Classification**: Categorizes items into non-toxic, mildly-toxic, highly-toxic, and unidentified
- **Intelligent Path Planning**: Implements PSO (Particle Swarm Optimization) algorithm for optimal robotic arm movement
- **Dual Camera System**: Supports multiple camera inputs for comprehensive detection
- **GUI Interface**: User-friendly Tkinter-based interface for system control and monitoring
- **Automated Sorting**: Robotic arm automatically sorts items into appropriate bins based on toxicity levels
- **Data Logging**: Comprehensive logging of detection results and sorting operations
- **Report Generation**: Generates toxicity reports and statistics

## 🏗️ System Architecture

### Core Components

- **`latestArmControl.py`**: Main control system integrating detection, path planning, and robotic arm control
- **`gui_app.py`**: Graphical user interface for system monitoring and control
- **`robotic_arm.py`**: Robotic arm hardware interface and control
- **`path_planning/`**: PSO-based path planning algorithms
  - `pso.py`: Particle Swarm Optimization implementation
  - `environment.py`: Environment modeling and validation
- **`shared_state.py`**: Shared state management for multi-threaded operations
- **`report_generator.py`**: Report generation and data visualization

### Detection and Classification

The system uses YOLO models trained on electronic waste datasets to:
- Detect electronic components in real-time
- Classify items by toxicity level:
  - 🟢 **Non-toxic** (Green bin)
  - 🟡 **Mildly-toxic** (Yellow bin)  
  - 🔴 **Highly-toxic** (Red bin)
  - ⚪ **Unidentified** (Gray bin)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- Tkinter (usually included with Python)
- NumPy
- Pandas
- PIL (Pillow)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd pso
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install opencv-python ultralytics numpy pandas pillow
   ```

4. **Run the system:**
   ```bash
   python gui_app.py
   ```

## 🎮 Usage

### GUI Interface

1. **Launch the application:**
   ```bash
   python gui_app.py
   ```

2. **Camera Setup**: The system will automatically detect and initialize available cameras

3. **Detection**: Items detected by the camera will be automatically classified and queued for sorting

4. **Manual Control**: Use the GUI controls to:
   - Start/stop detection
   - Control robotic arm movement
   - View real-time camera feeds
   - Monitor sorting statistics

### Programmatic Usage

```python
from latestArmControl import DetectionData
from path_planning.pso import PSO
from robotic_arm import RoboticArm

# Initialize components
arm = RoboticArm()
pso = PSO()

# Start detection and sorting
detection_data = DetectionData()
# ... detection and sorting logic
```

## 📁 Project Structure

```
pso/
├── latestArmControl.py      # Main control system
├── gui_app.py              # GUI interface
├── robotic_arm.py         # Robotic arm control
├── shared_state.py        # Shared state management
├── report_generator.py    # Report generation
├── path_planning/         # PSO path planning
│   ├── pso.py
│   └── environment.py
├── without PSO/           # Alternative implementation
└── README.md
```

## 🔧 Configuration

### Camera Settings
- Primary camera: `/dev/video1`
- Secondary camera: `/dev/video2`
- Resolution: Configurable via GUI

### Robotic Arm Configuration
- Servo angles for different bins
- Movement speed and precision settings
- Safety limits and constraints

### PSO Parameters
- Population size
- Iteration limits
- Convergence criteria
- Environment constraints

## 📊 Data and Logging

The system generates several types of data:

- **Detection Logs**: JSON files containing detection results
- **Summary Reports**: Excel files with sorting statistics
- **Toxicity Reports**: Visual graphs and charts
- **Unidentified Items**: Logs for items that couldn't be classified

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**: Check camera permissions and device paths
2. **YOLO model errors**: Ensure model files are in the correct directory
3. **Robotic arm connection**: Verify hardware connections and drivers
4. **GUI not displaying**: Check Tkinter installation and display settings

### Support

For issues and questions:
- Check the troubleshooting section
- Review the code comments
- Open an issue on GitHub

## 🔮 Future Enhancements

- [ ] Machine learning model improvements
- [ ] Additional camera support
- [ ] Web-based interface
- [ ] Database integration
- [ ] Mobile app companion
- [ ] Advanced path optimization algorithms

---

**Note**: This system is designed for educational and research purposes. Ensure proper safety protocols when operating robotic equipment.

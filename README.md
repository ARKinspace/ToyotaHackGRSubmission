# RaceTrack Studio

IF YOU JUST WANT A ONE-CLICK EXECUTABLE WITH SAME FUNCTIONALITY WITHOUT SETTING UP THE .VENV AND INSTALLING THE REQUIRED PACKAGES - FOLLOW THIS LINK: 
https://drive.google.com/file/d/1vqKAbIPMqP4TIQTAbmfHB6MgAO_Abdf1/view?usp=sharing

A comprehensive race telemetry analysis and visualization tool for Toyota GR Cup racing data.

## Features

### 🗺️ Track Scanning & Creation
- **GPS-based Track Generation**: Automatically generate race track maps from GPS telemetry data
- **Manual Fine-Tuning**: Adjust track layouts with precision editing tools
- **Elevation Integration**: Fetch real-world elevation data for accurate 3D track representation
- **Flat Track Option**: Skip elevation fetching for faster processing

### 📊 Telemetry Analysis
- **Multi-Session Support**: Load and analyze Race 1 and Race 2 data simultaneously
- **Vehicle Comparison**: Compare telemetry across multiple vehicles/drivers
- **Comprehensive Data**: Speed, acceleration, steering, braking, GPS coordinates, and more
- **Session Persistence**: Save and load complete analysis sessions (.sark files)

### 🏁 Optimal Racing Line
- **Physics-Based Calculation**: Weather-adjusted optimal line using vehicle dynamics
- **Temperature Effects**: Grip coefficients adjust based on track temperature
- **Rain Conditions**: Support for dry, damp, intermediate, and wet weather
- **3D Visualization**: Golden racing line displayed with accurate elevation following

### 🎨 3D Track Rendering
- **Interactive 3D View**: Rotate, zoom, and pan around the track
- **Telemetry Overlay**: Visualize speed, acceleration, and other metrics on the track
- **Optimal Line Display**: See the calculated optimal racing line in golden color
- **Elevation Profiles**: Accurate 3D representation of track elevation changes

## Installation

### Prerequisites
- Python 3.10 or higher
- Windows, macOS, or Linux

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ARKinReplication/ToyotaHackGR.git
   cd ToyotaHackGR
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Launch the Application

```bash
python main.py
```

### Workflow

1. **Track Scanner (Tab 1)**
   - Select a dataset folder containing telemetry files
   - The track will be automatically generated from GPS data
   - Optionally enable "Flat Track" to skip elevation fetching
   - Click "Finalize Track" to calculate optimal racing line and proceed

2. **Fine Tuner (Tab 2)**
   - Adjust track layout if needed
   - Modify control points for precision
   - Finalize adjustments

3. **3D Render (Tab 3)**
   - View the track in interactive 3D
   - See the golden optimal racing line
   - Explore elevation changes

4. **Race Telemetry (Tab 4)**
   - Load telemetry data for specific vehicles/sessions
   - Analyze lap times, speed profiles, and more
   - Compare multiple drivers

### Session Management

- **Save Session**: File → Save Session (saves to `.sark` file)
- **Load Session**: File → Load Session (restores complete state)

## Supported Datasets

The application works with Toyota GR Cup race data with the following structure:

```
dataset-folder/
├── R1_track_telemetry_data.csv      # Race 1 telemetry
├── R2_track_telemetry_data.csv      # Race 2 telemetry
├── 26_Weather_Race 1.CSV            # Weather data
├── 26_Weather_Race 2.CSV
└── ...other race files
```

### Dataset GPS Availability

- ✅ **Barber Motorsports Park**: Full GPS support
- ✅ **Virginia International Raceway**: Full GPS support
- ❌ **Road America**: No GPS data
- ❌ **Sebring**: No GPS data
- ❌ **Circuit of the Americas (COTA)**: No GPS data
- ❌ **Indianapolis Motor Speedway**: No GPS data
- ❌ **Sonoma Raceway**: No GPS data

*Note: Datasets without GPS can still be analyzed for telemetry, but track generation is limited.*

## Technology Stack

- **GUI Framework**: PyQt6
- **3D Graphics**: PyQtGraph with OpenGL
- **Data Processing**: Pandas, NumPy
- **Scientific Computing**: SciPy (spline interpolation, spatial KD-trees)
- **Machine Learning**: scikit-learn
- **Computer Vision**: OpenCV
- **Imaging**: Pillow, ImageIO
- **Visualization**: Matplotlib

## Project Structure

```
ToyotaHackGR/
├── main.py                          # Application entry point
├── Code/
│   ├── Core/
│   │   ├── MapCreator/              # Track generation
│   │   ├── OptimalLine/             # Optimal line calculation
│   │   ├── ReadEngine/              # Telemetry reading
│   │   └── TelemetryEngine/         # Telemetry processing
│   └── GUI/
│       ├── MainWindow.py            # Main application window
│       ├── TrackScanner.py          # Tab 1: Track scanning
│       ├── FineTuner.py             # Tab 2: Fine tuning
│       ├── Render3D.py              # Tab 3: 3D visualization
│       └── RaceTelemetryTab.py      # Tab 4: Telemetry analysis
├── Non_Code/
│   └── dataSets/                    # Race data (gitignored)
├── outputs/                         # Generated files (gitignored)
└── tests/                           # Test scripts
```

## Features in Detail

### Optimal Racing Line

The optimal line calculation considers:
- **Vehicle Physics**: Mass, drag coefficient, tire grip (μ)
- **Track Temperature**: Optimal grip at 85°C, degrades with deviation
- **Weather Conditions**: Dry (μ=1.40) to Wet (μ=0.70)
- **Cornering Forces**: Maximum lateral G-forces
- **Braking Performance**: Deceleration capabilities
- **Speed Optimization**: Entry/exit speed calculations

### Weather Integration

Automatically parses weather data:
- Air temperature
- Track temperature (estimated as air temp + 10°C)
- Rainfall levels
- Humidity
- Wind speed

## Contributing

This project was developed for the Toyota GR Hackathon. Contributions, suggestions, and improvements are welcome!

## License

See LICENSE file for details.

## Acknowledgments

- Toyota Gazoo Racing for the GR Cup dataset
- PyQt6 and PyQtGraph teams for excellent GUI frameworks
- SciPy and NumPy communities for scientific computing tools

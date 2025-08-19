import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

# =============================================
# 1. LOAD TRAINED MODEL
# =============================================

def load_aqi_model(model_filename='AQI.joblib'):
    """Load the trained AQI model"""
    try:
        if not os.path.exists(model_filename):
            print(f" Model file '{model_filename}' not found!")
            print("Available .joblib files:")
            joblib_files = [f for f in os.listdir('.') if f.endswith('.joblib')]
            for file in joblib_files:
                print(f"  - {file}")
            return None
        
        model_data = joblib.load(model_filename)
        model_pipeline = model_data['model_pipeline']
        
        print(f" Model loaded successfully!")
        print(f" Model Info:")
        print(f"  Algorithm: {model_data['training_info']['algorithm']}")
        print(f"  Test R²: {model_data['performance_metrics']['test_r2']:.4f}")
        print(f"  Test MAE: {model_data['performance_metrics']['test_mae']:.4f}")
        print(f"  Available Stations: {model_data['data_info']['stations']}")
        
        return model_data, model_pipeline
        
    except Exception as e:
        print(f" Error loading model: {e}")
        return None

# =============================================
# 2. PREDICTION FUNCTIONS
# =============================================

def get_aqi_category(aqi):
    """Convert AQI value to category with emoji and color"""
    if aqi <= 50:
        return "Good 🟢", "#00E400"
    elif aqi <= 100:
        return "Satisfactory 🟡", "#FFFF00"
    elif aqi <= 200:
        return "Moderate 🟠", "#FF7E00"
    elif aqi <= 300:
        return "Poor 🔴", "#FF0000"
    elif aqi <= 400:
        return "Very Poor 🟣", "#8F3F97"
    else:
        return "Severe ⚫", "#7E0023"

def get_health_advice(category):
    """Get detailed health advice based on AQI category"""
    advice_map = {
        "Good ": {
            "advice": "Air quality is good. Perfect for all outdoor activities.",
            "activities": "Jogging, cycling, outdoor sports recommended",
            "precautions": "No precautions needed"
        },
        "Satisfactory 🟡": {
            "advice": "Air quality is acceptable for most people.",
            "activities": "Normal outdoor activities are fine",
            "precautions": "Sensitive individuals may experience minor discomfort"
        },
        "Moderate 🟠": {
            "advice": "Unhealthy for sensitive groups.",
            "activities": "Reduce prolonged outdoor exertion",
            "precautions": "People with respiratory conditions should limit outdoor activities"
        },
        "Poor 🔴": {
            "advice": "Everyone should limit outdoor activities.",
            "activities": "Avoid outdoor sports and exercise",
            "precautions": "Wear masks when going outside, keep windows closed"
        },
        "Very Poor 🟣": {
            "advice": "Avoid outdoor activities entirely.",
            "activities": "Stay indoors, use air purifiers",
            "precautions": "Serious health risk for everyone, especially children and elderly"
        },
        "Severe ⚫": {
            "advice": "Health emergency! Stay indoors.",
            "activities": "Emergency conditions - no outdoor activities",
            "precautions": "Seek immediate medical attention if experiencing breathing difficulties"
        }
    }
    return advice_map.get(category, {
        "advice": "Unknown category",
        "activities": "Consult health guidelines",
        "precautions": "Exercise caution"
    })

def predict_aqi(model_pipeline, station_id, pm2_5, pm10, no2, so2, co, o3, temp_c, rain_mm):
    """Make AQI prediction with comprehensive results"""
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Station_ID': [station_id],
        'PM2_5': [pm2_5],
        'PM10': [pm10],
        'NO2': [no2],
        'SO2': [so2],
        'CO': [co],
        'O3': [o3],
        'Temp_C': [temp_c],
        'Rain_mm': [rain_mm]
    })
    
    try:
        # Make prediction
        predicted_aqi = model_pipeline.predict(input_data)[0]
        category, color = get_aqi_category(predicted_aqi)
        health_info = get_health_advice(category)
        
        return {
            'success': True,
            'station_id': station_id,
            'predicted_aqi': round(predicted_aqi, 2),
            'category': category,
            'color': color,
            'health_advice': health_info['advice'],
            'activities': health_info['activities'],
            'precautions': health_info['precautions'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_data': input_data.iloc[0].to_dict()
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# =============================================
# 3. USER INPUT FUNCTIONS
# =============================================

def get_user_input(available_stations):
    """Get environmental parameters from user input"""
    
    print("\n🌍 Enter Environmental Parameters for AQI Prediction")
    print("=" * 55)
    
    try:
        # Station ID
        print(f"\n🏭 Available Stations: {available_stations}")
        while True:
            try:
                station_id = int(input("Enter Station ID: "))
                if station_id in available_stations:
                    break
                else:
                    print(f"⚠️  Station {station_id} not in training data. Available: {available_stations}")
                    choice = input("Continue anyway? (y/n): ").lower()
                    if choice == 'y':
                        break
            except ValueError:
                print("❌ Please enter a valid integer for Station ID")
        
        # Pollutant concentrations
        print("\n💨 Pollutant Concentrations:")
        pm2_5 = float(input("PM2.5 (µg/m³) [0-500]: "))
        pm10 = float(input("PM10 (µg/m³) [0-600]: "))
        no2 = float(input("NO2 (µg/m³) [0-200]: "))
        so2 = float(input("SO2 (µg/m³) [0-100]: "))
        co = float(input("CO (mg/m³) [0-10]: "))
        o3 = float(input("O3 (µg/m³) [0-300]: "))
        
        # Weather conditions
        print("\n🌤️  Weather Conditions:")
        temp_c = float(input("Temperature (°C) [-10 to 50]: "))
        rain_mm = float(input("Rainfall (mm) [0-50]: "))
        
        # Validation
        if not (0 <= pm2_5 <= 500):
            print("⚠️  PM2.5 value seems unusual (typical range: 0-500)")
        if not (0 <= pm10 <= 600):
            print("⚠️  PM10 value seems unusual (typical range: 0-600)")
        if not (-10 <= temp_c <= 50):
            print("⚠️  Temperature seems unusual (typical range: -10 to 50°C)")
        
        return {
            'station_id': station_id,
            'pm2_5': pm2_5,
            'pm10': pm10,
            'no2': no2,
            'so2': so2,
            'co': co,
            'o3': o3,
            'temp_c': temp_c,
            'rain_mm': rain_mm
        }
    
    except ValueError:
        print("❌ Invalid input. Please enter numeric values only.")
        return None
    except KeyboardInterrupt:
        print("\n👋 Prediction cancelled by user.")
        return None

def get_preset_scenarios():
    """Predefined scenarios for quick testing"""
    scenarios = {
        '1': {
            'name': 'Clean Air Day',
            'station_id': 1,
            'pm2_5': 15, 'pm10': 25, 'no2': 10, 'so2': 5,
            'co': 0.3, 'o3': 45, 'temp_c': 22, 'rain_mm': 2
        },
        '2': {
            'name': 'Moderate Pollution',
            'station_id': 1,
            'pm2_5': 65, 'pm10': 85, 'no2': 30, 'so2': 15,
            'co': 1.2, 'o3': 75, 'temp_c': 28, 'rain_mm': 0
        },
        '3': {
            'name': 'Heavy Pollution Day',
            'station_id': 1,
            'pm2_5': 150, 'pm10': 220, 'no2': 60, 'so2': 35,
            'co': 2.5, 'o3': 120, 'temp_c': 32, 'rain_mm': 0
        },
        '4': {
            'name': 'Post-Rain Clear',
            'station_id': 2,
            'pm2_5': 25, 'pm10': 40, 'no2': 12, 'so2': 6,
            'co': 0.4, 'o3': 55, 'temp_c': 24, 'rain_mm': 15
        }
    }
    return scenarios

# =============================================
# 4. DISPLAY FUNCTIONS
# =============================================

def display_prediction_results(result):
    """Display prediction results in a formatted way"""
    
    if not result['success']:
        print(f"❌ Prediction failed: {result['error']}")
        return
    
    print("\n" + "="*60)
    print("🎯 AQI PREDICTION RESULTS")
    print("="*60)
    
    # Basic info
    print(f"\n📍 Station: {result['station_id']}")
    print(f"🕐 Timestamp: {result['timestamp']}")
    
    # Main prediction
    print(f"\n🎯 PREDICTED AQI: {result['predicted_aqi']}")
    print(f"📊 Category: {result['category']}")
    
    # Health information
    print(f"\n💡 Health Advice:")
    print(f"   {result['health_advice']}")
    print(f"\n🏃 Activities: {result['activities']}")
    print(f"⚠️  Precautions: {result['precautions']}")
    
    # Input summary
    print(f"\n📋 Input Parameters Used:")
    input_data = result['input_data']
    print(f"   PM2.5: {input_data['PM2_5']} µg/m³")
    print(f"   PM10: {input_data['PM10']} µg/m³")
    print(f"   NO2: {input_data['NO2']} µg/m³")
    print(f"   SO2: {input_data['SO2']} µg/m³")
    print(f"   CO: {input_data['CO']} mg/m³")
    print(f"   O3: {input_data['O3']} µg/m³")
    print(f"   Temperature: {input_data['Temp_C']}°C")
    print(f"   Rainfall: {input_data['Rain_mm']} mm")

# =============================================
# 5. BATCH PREDICTION FROM FILE
# =============================================

def batch_prediction_from_csv(model_pipeline, csv_filename):
    """Make predictions for multiple entries from CSV file"""
    
    try:
        # Load CSV
        data = pd.read_csv(csv_filename)
        print(f"✅ Loaded {len(data)} records from {csv_filename}")
        
        # Check required columns
        required_cols = ['Station_ID', 'PM2_5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temp_C', 'Rain_mm']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return
        
        # Make predictions
        predictions = model_pipeline.predict(data[required_cols])
        
        # Add results to dataframe
        data['Predicted_AQI'] = predictions.round(2)
        data['AQI_Category'] = data['Predicted_AQI'].apply(lambda x: get_aqi_category(x)[0])
        
        # Save results
        output_filename = f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        data.to_csv(output_filename, index=False)
        
        print(f"✅ Predictions saved to: {output_filename}")
        print(f"📊 Results summary:")
        print(data['AQI_Category'].value_counts())
        
        return data
        
    except FileNotFoundError:
        print(f"❌ File {csv_filename} not found")
    except Exception as e:
        print(f"❌ Error in batch prediction: {e}")

# =============================================
# 6. MAIN INTERACTIVE FUNCTION
# =============================================

def main():
    """Main interactive AQI prediction function"""
    
    print("🌟 AQI PREDICTION SYSTEM")
    print("=" * 50)
    print("Welcome to the AI-powered Air Quality Index predictor!")
    
    # Load model
    model_data = load_aqi_model()
    if model_data is None:
        return
    
    model_info, model_pipeline = model_data
    available_stations = model_info['data_info']['stations']
    
    while True:
        print("\n🔍 Choose prediction method:")
        print("1. 📝 Enter custom parameters")
        print("2. 🎭 Use preset scenarios")
        print("3. 📁 Batch prediction from CSV")
        print("4. ℹ️  Show model information")
        # print("5. 👋 Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # Custom input
            user_input = get_user_input(available_stations)
            if user_input:
                result = predict_aqi(model_pipeline, **user_input)
                display_prediction_results(result)
        
        elif choice == '2':
            # Preset scenarios
            scenarios = get_preset_scenarios()
            print("\n🎭 Available Scenarios:")
            for key, scenario in scenarios.items():
                print(f"{key}. {scenario['name']}")
            
            scenario_choice = input("Choose scenario (1-4): ").strip()
            if scenario_choice in scenarios:
                scenario = scenarios[scenario_choice]
                print(f"\n🎬 Running scenario: {scenario['name']}")
                
                # Remove name from parameters
                params = {k: v for k, v in scenario.items() if k != 'name'}
                result = predict_aqi(model_pipeline, **params)
                display_prediction_results(result)
            else:
                print("❌ Invalid scenario choice")
        
        elif choice == '3':
            # Batch prediction
            csv_file = input("Enter CSV filename: ").strip()
            batch_prediction_from_csv(model_pipeline, csv_file)
        
        elif choice == '4':
            # Model information
            print("\n📊 MODEL INFORMATION")
            print("-" * 30)
            training_info = model_info['training_info']
            performance = model_info['performance_metrics']
            
            print(f"Algorithm: {training_info['algorithm']}")
            print(f"Created: {training_info['created_date']}")
            print(f"Training samples: {training_info['train_samples']}")
            print(f"Test samples: {training_info['test_samples']}")
            print(f"Features: {training_info['total_features']}")
            print(f"Stations: {training_info['unique_stations']}")
            print(f"\nPerformance:")
            print(f"  Test R²: {performance['test_r2']:.4f}")
            print(f"  Test MAE: {performance['test_mae']:.4f}")
            print(f"  Test RMSE: {performance['test_rmse']:.4f}")
            print(f"  CV Score: {performance['cv_mean']:.4f} ± {performance['cv_std']:.4f}")
        
        elif choice == '5':
            print("👋 Thank you for using AQI Prediction System!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-5.")
        
        # Ask if user wants to continue
        if choice in ['1', '2', '3']:
            continue_choice = input("\n🔄 Make another prediction? (y/n): ").lower()
            if continue_choice != 'y':
                print("👋 Thank you for using AQI Prediction System!")
                break

# =============================================
# 7. COMMAND LINE VERSION
# =============================================

def quick_predict():
    """Quick single prediction from command line arguments"""
    import sys
    
    if len(sys.argv) != 10:
        print("Usage: python aqi_predict.py <station_id> <pm2.5> <pm10> <no2> <so2> <co> <o3> <temp> <rain>")
        print("Example: python aqi_predict.py 1 45 68 22 8 0.8 35 28 0")
        return
    
    try:
        # Parse command line arguments
        _, station_id, pm2_5, pm10, no2, so2, co, o3, temp_c, rain_mm = sys.argv
        
        params = {
            'station_id': int(station_id),
            'pm2_5': float(pm2_5),
            'pm10': float(pm10),
            'no2': float(no2),
            'so2': float(so2),
            'co': float(co),
            'o3': float(o3),
            'temp_c': float(temp_c),
            'rain_mm': float(rain_mm)
        }
        
        # Load model and predict
        model_data = load_aqi_model()
        if model_data is None:
            return
        # @route('/pm_2_5')
        model_info, model_pipeline = model_data
        result = predict_aqi(model_pipeline, **params)
        
        if result['success']:
            print(f"AQI: {result['predicted_aqi']} ({result['category']})")
        else:
            print(f"Error: {result['error']}")
            
    except ValueError:
        print("❌ Invalid parameters. Please check your input values.")
    except Exception as e:
        print(f"❌ Error: {e}")

# =============================================
# 8. RUN THE PROGRAM
# =============================================

if __name__ == "__main__":
    import sys
    
    # Check if command line arguments provided for quick prediction
    if len(sys.argv) > 1:
        quick_predict()
    else:
        main()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import datetime
import os

# =============================================
# 1. LOAD AND PREPARE DATA WITH STATION_ID
# =============================================


data = pd.read_csv("air.csv")

# Drop unnecessary columns but KEEP Station_ID
data = data.drop(columns=['DateTime','Humidity','Wind_Speed','Wind_Direction','Pressure_hPa'])

# Include Station_ID in features
X = data[['Station_ID', 'PM2_5','PM10','NO2','SO2','CO','O3','Temp_C','Rain_mm']]
y = data['AQI_Target']

print(f"📊 Dataset Info:")
print(f"Total samples: {len(data)}")
print(f"Features: {X.columns.tolist()}")
print(f"Unique stations: {data['Station_ID'].nunique()}")
print(f"Station distribution:\n{data['Station_ID'].value_counts()}")

# Check for missing values
print(f"\n🔍 Missing values check:")
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])

# Remove rows with missing values if any
if data.isnull().sum().sum() > 0:
    print(f"Removing {data.isnull().any(axis=1).sum()} rows with missing values...")
    data = data.dropna()
    X = data[['Station_ID', 'PM2_5','PM10','NO2','SO2','CO','O3','Temp_C','Rain_mm']]
    y = data['AQI_Target']
    print(f"Final dataset shape: {data.shape}")

# =============================================
# 2. DATA EXPLORATION
# =============================================

print("\n📈 Data Exploration:")
print(f"AQI Target Statistics:")
print(f"  Mean: {y.mean():.2f}")
print(f"  Std: {y.std():.2f}")
print(f"  Min: {y.min():.2f}")
print(f"  Max: {y.max():.2f}")

# Station-wise statistics
station_stats = data.groupby('Station_ID')['AQI_Target'].agg(['mean', 'std', 'count'])
print(f"\n🏭 Station-wise AQI Statistics:")
print(station_stats)

# =============================================
# 3. TRAIN MODEL WITH STATION_ID
# =============================================

print(f"\n🎯 Training Random Forest Model with Station_ID...")

# Proper train-test split with stratification by Station_ID
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=X['Station_ID']
)

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Preprocessing pipeline for mixed data types
numeric_features = ['PM2_5','PM10','NO2','SO2','CO','O3','Temp_C','Rain_mm']
categorical_features = ['Station_ID']

preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
], remainder='drop')

# Create pipeline with preprocessing + Random Forest
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    ))
])

# Train the model
print("Training Random Forest...")
model_pipeline.fit(X_train, y_train)
print("✅ Model training completed!")

# =============================================
# 4. MODEL EVALUATION
# =============================================

print(f"\n📊 Model Evaluation...")

# Make predictions
y_test_pred = model_pipeline.predict(X_test)
y_train_pred = model_pipeline.predict(X_train)

# Calculate performance metrics
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("\n🎯 Model Performance:")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

print(f"\n📚 Training Performance:")
print(f"Train R² Score: {train_r2:.4f}")
print(f"Train MAE: {train_mae:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")

# Check for overfitting
print(f"\n🔍 Overfitting Check:")
r2_diff = train_r2 - test_r2
mae_diff = test_mae - train_mae

print(f"R² difference (train - test): {r2_diff:.4f}")
print(f"MAE difference (test - train): {mae_diff:.4f}")

if r2_diff > 0.1:
    print("⚠️  High R² difference suggests potential overfitting")
elif r2_diff > 0.05:
    print("⚠️  Moderate R² difference - monitor for overfitting")
else:
    print("✅ Low R² difference - good generalization")

# Cross-validation for robust evaluation
print(f"\n🔄 Cross-Validation (5-fold)...")
cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
print(f"CV R² scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"Average CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# =============================================
# 5. FEATURE IMPORTANCE ANALYSIS
# =============================================

print(f"\n📊 Feature Importance Analysis...")

# Get feature importance from trained model
rf_model = model_pipeline.named_steps['regressor']

# Get feature names after preprocessing
ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
feature_names_transformed = numeric_features + list(ohe_feature_names)

feature_importance = pd.DataFrame({
    'feature': feature_names_transformed,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📈 Feature Importance Rankings:")
print(feature_importance)

# Top 5 most important features
print(f"\n🏆 Top 5 Most Important Features:")
for i, (_, row) in enumerate(feature_importance.head().iterrows(), 1):
    print(f"{i}. {row['feature']}: {row['importance']:.4f}")

# =============================================
# 6. SAVE MODEL
# =============================================

print(f"\n💾 Saving Model...")

# Prepare model data with comprehensive metadata
model_data = {
    'model_pipeline': model_pipeline,
    'feature_names': list(X.columns),
    'feature_names_transformed': feature_names_transformed,
    'performance_metrics': {
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'overfitting_check': {
            'r2_diff': r2_diff,
            'mae_diff': mae_diff
        }
    },
    'training_info': {
        'algorithm': 'RandomForestRegressor with Station_ID',
        'n_estimators': 100,
        'random_state': 42,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'total_features': len(X.columns),
        'unique_stations': data['Station_ID'].nunique(),
        'created_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    },
    'feature_importance': feature_importance.to_dict('records'),
    'preprocessing_info': {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'encoding': 'OneHotEncoder',
        'strategy': 'drop_first'
    },
    'station_stats': station_stats.to_dict('index'),
    'data_info': {
        'total_samples': len(data),
        'stations': sorted(data['Station_ID'].unique()),
        'aqi_range': [float(y.min()), float(y.max())],
        'aqi_mean': float(y.mean())
    }
}

# Save model
model_filename = 'AQI.joblib'
joblib.dump(model_data, model_filename)

print(f"✅ Model saved successfully as: {model_filename}")
if os.path.exists(model_filename):
    file_size_kb = round(os.path.getsize(model_filename) / 1024, 2)
    print(f"📁 File size: {file_size_kb} KB")

# =============================================
# 7. PREDICTION FUNCTIONS
# =============================================

def predict_aqi_with_station(station_id, pm2_5, pm10, no2, so2, co, o3, temp_c, rain_mm):
    """
    Predict AQI including station location effects
    
    Parameters:
    - station_id: Station ID number
    - pm2_5: PM2.5 concentration (µg/m³)
    - pm10: PM10 concentration (µg/m³)  
    - no2: NO2 concentration (µg/m³)
    - so2: SO2 concentration (µg/m³)
    - co: CO concentration (mg/m³)
    - o3: O3 concentration (µg/m³)
    - temp_c: Temperature (°C)
    - rain_mm: Rainfall (mm)
    
    Returns:
    - Dictionary with prediction results
    """
    
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
    
    predicted_aqi = model_pipeline.predict(input_data)[0]
    category = get_aqi_category(predicted_aqi)
    
    return {
        'station_id': station_id,
        'predicted_aqi': round(predicted_aqi, 2),
        'category': category,
        'health_advice': get_health_advice(category),
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def get_aqi_category(aqi):
    """Convert AQI value to category with emoji"""
    if aqi <= 50:
        return "Good 🟢"
    elif aqi <= 100:
        return "Satisfactory 🟡"
    elif aqi <= 200:
        return "Moderate 🟠"
    elif aqi <= 300:
        return "Poor 🔴"
    elif aqi <= 400:
        return "Very Poor 🟣"
    else:
        return "Severe ⚫"

def get_health_advice(category):
    """Get health advice based on AQI category"""
    advice_map = {
        "Good 🟢": "Air quality is good. Ideal for outdoor activities.",
        "Satisfactory 🟡": "Air quality is acceptable for most people.",
        "Moderate 🟠": "Sensitive individuals should consider limiting outdoor activities.",
        "Poor 🔴": "Everyone should limit outdoor activities. Wear a mask if going outside.",
        "Very Poor 🟣": "Avoid outdoor activities. Stay indoors with air purification.",
        "Severe ⚫": "Health emergency. Avoid all outdoor activities."
    }
    return advice_map.get(category, "Unknown category")

# =============================================
# 8. EXAMPLE PREDICTIONS
# =============================================

print(f"\n🔮 Example Predictions:")

# Test conditions - moderate pollution scenario
test_conditions = {
    'pm2_5': 45, 'pm10': 68, 'no2': 22, 'so2': 8,
    'co': 0.8, 'o3': 35, 'temp_c': 28, 'rain_mm': 0
}

print(f"\nTest conditions: {test_conditions}")

# Get unique stations for predictions
unique_stations = sorted(data['Station_ID'].unique())
print(f"\n🏭 Station-Specific Predictions:")

for station in unique_stations[:3]:  # Show first 3 stations
    result = predict_aqi_with_station(station, **test_conditions)
    print(f"\nStation {result['station_id']}:")
    print(f"  Predicted AQI: {result['predicted_aqi']}")
    print(f"  Category: {result['category']}")
    print(f"  Health Advice: {result['health_advice']}")

# =============================================
# 9. MODEL SUMMARY
# =============================================

print(f"\n" + "="*60)
print("📊 MODEL TRAINING SUMMARY")
print("="*60)

print(f"\n🎯 Final Model Performance:")
print(f"  Test R² Score: {test_r2:.4f} ({test_r2*100:.1f}% variance explained)")
print(f"  Test MAE: {test_mae:.4f} AQI points")
print(f"  Test RMSE: {test_rmse:.4f} AQI points")
print(f"  Cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print(f"\n🏆 Model Quality Assessment:")
if test_r2 >= 0.9:
    print("  ✅ Excellent performance (R² ≥ 0.90)")
elif test_r2 >= 0.8:
    print("  ✅ Good performance (R² ≥ 0.80)")
elif test_r2 >= 0.7:
    print("  ⚠️  Fair performance (R² ≥ 0.70)")
else:
    print("  ❌ Poor performance (R² < 0.70)")

if test_mae <= 5:
    print("  ✅ Excellent accuracy (MAE ≤ 5 AQI points)")
elif test_mae <= 10:
    print("  ✅ Good accuracy (MAE ≤ 10 AQI points)")
else:
    print("  ⚠️  Fair accuracy (MAE > 10 AQI points)")

print(f"\n📁 Saved Files:")
print(f"  Model: {model_filename}")
print(f"  Size: {file_size_kb} KB")

print(f"\n🚀 Model is ready for deployment!")
print(f"   Use predict_aqi_with_station() function for predictions")

print(f"\n✨ Training completed successfully at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

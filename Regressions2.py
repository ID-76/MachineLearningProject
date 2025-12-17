"""
COMPARING HUBER REGRESSION vs RIDGE REGRESSION
================================================

WHY COMPARE THESE TWO?
---------------------
Both are designed to handle PROBLEMATIC DATA, but in different ways:

HUBER REGRESSION:
- Problem it solves: OUTLIERS (extreme values that don't fit the pattern)
- How it works: Uses a "loss function" that is less sensitive to outliers
- Think of it as: A teacher who doesn't let one bad grade ruin your average
- Best for: Data with weird extreme values (like 10-hour traffic accidents!)

RIDGE REGRESSION:
- Problem it solves: MULTICOLLINEARITY (when features are too similar/correlated)
- How it works: Adds a penalty (L2 regularization) to prevent overfitting
- Think of it as: A teacher who prevents you from memorizing answers
- Best for: Data with many correlated features (like temperature and wind_chill)

Let's see which one works better for our traffic accident data!
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor, Ridge  # Our two models!
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# import matplotlib.pyplot as plt
# import seaborn as sns

# Set style for prettier plots
#plt.style.use('seaborn-v0_8-darkgrid')

# STEP 1: LOAD THE DATA FROM CSV FILE
# Load our CSV file
df = pd.read_csv("Data/US_Accidents_processed_for_modeling.csv")

print(f"Data loaded successfully!")

# Check for any missing values
print("\nChecking for missing values...")
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print("WARNING: Found missing values in these columns:")
    print(missing_values[missing_values > 0])
else:
    print("No missing values found!")

# STEP 2: SELECT FEATURES AND TARGET

# What are we predicting?
target = 'duration_min'

# What information are we using? These features!
features = [
    'Humidity(%)',          # How humid the air was
    'start_hour',           # What time the accident happened
    'start_dayofweek',      # What day of the week
    'start_month',          # What month
    'Temperature(C)',       # Temperature in Celsius
    'Wind_Speed(m/s)',      # Wind speed in meters per second
    'Visibility(km)',       # How far you could see
    'Precipitation(mm)',    # Amount of rain/snow
    'Pressure(hPa)',        # Atmospheric pressure
    'Distance(km)',         # Length of road affected
    'Severity'              # How severe (1-4 scale)
]

# Create our input (X) and output (y) data
X = df[features]
y = df[target]

print(f"Dataset loaded: {len(df)} accidents")
# STEP 3: SPLIT THE DATA

# Why split?
# TRAINING SET (80%): We teach the models using this data
# TESTING SET (20%): We evaluate how well they learned on NEW data they've never seen

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,        # 20% for testing
    random_state=42       # Makes results reproducible (same split every time)
)

print("Data Split done")

# STEP 4: SCALE THE FEATURES

# Why scale?
# Different features have VERY different scales:
# - Pressure: around 1000 hPa
# - Severity: just 1-4
# 
# Without scaling, features with bigger numbers would dominate!
# Scaling makes all features have similar importance (mean=0, std=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Learn scaling from training data
X_test_scaled = scaler.transform(X_test)        # Apply same scaling to test data

print("Features scaled successfully!")

# STEP 5: TRAIN BOTH MODELS

print("TRAINING BOTH REGRESSION MODELS")

# MODEL 1: HUBER REGRESSION (Robust to outliers)

print("\n MODEL 1: HUBER REGRESSION")
# Purpose: Handle OUTLIERS (extreme values)
#How: Uses a special 'loss function' that ignores extreme errors

huber = HuberRegressor(
    epsilon=1.35,        # Controls outlier tolerance (1.35 is standard)
                         # Lower epsilon = less tolerant of outliers
                         # Higher epsilon = more tolerant
    max_iter=1000,       # Maximum training iterations
    alpha=0.0001,        # Small regularization to prevent overfitting
)

huber.fit(X_train_scaled, y_train)
print("Huber Regression trained!")

# MODEL 2: RIDGE REGRESSION (Handles multicollinearity)

print("\nMODEL 2: RIDGE REGRESSION")
print("Purpose: Handle MULTICOLLINEARITY (correlated features)")
print("How: Adds L2 penalty to prevent coefficients from getting too large")

ridge = Ridge(
    alpha=1.0,           # Regularization strength (higher = more penalty)
                         # alpha=0 would be like regular linear regression
                         # alpha=1.0 is a good starting point
    random_state=42      # For reproducibility
)

ridge.fit(X_train_scaled, y_train)
print("Ridge Regression trained!")

print("\nBoth models successfully trained!")

# STEP 6: MAKE PREDICTIONS
print("MAKING PREDICTIONS ON TEST DATA")

# Now both models predict accident duration for the test set
y_pred_huber = huber.predict(X_test_scaled)
y_pred_ridge = ridge.predict(X_test_scaled)

print("Predictions completed for both models!")

# Show some example predictions
print("\nExample predictions (first 10 test cases):")
comparison_sample = pd.DataFrame({
    'Actual (min)': y_test.values[:10],
    'Huber Pred': y_pred_huber[:10],
    'Ridge Pred': y_pred_ridge[:10],
    'Huber Error': y_test.values[:10] - y_pred_huber[:10],
    'Ridge Error': y_test.values[:10] - y_pred_ridge[:10]
})
print(comparison_sample.to_string(index=False))

# STEP 7: EVALUATE BOTH MODELS
print("EVALUATING MODEL PERFORMANCE")

# Calculate metrics for HUBER
r2_huber = r2_score(y_test, y_pred_huber)
mae_huber = mean_absolute_error(y_test, y_pred_huber)
rmse_huber = np.sqrt(mean_squared_error(y_test, y_pred_huber))

# Calculate metrics for RIDGE
r2_ridge = r2_score(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

print("\nMETRIC EXPLANATIONS:")
print("1. R² Score (R-squared):") # Measures how well the model fits the data
print("   → Range: 0 to 1 (higher is better)")
print("   → 1.0 = Perfect predictions")
print("   → 0.5 = Model explains 50% of the variation")
print("   → 0.0 = Model is useless (just guessing the average)")
print()
print("2. MAE (Mean Absolute Error):") # Average difference between prediction and reality
print("   → In MINUTES for our case")
print("   → Lower is better")
print()
print("3. RMSE (Root Mean Squared Error):") # Similar to MAE but punishes BIG errors more
print("   → Also in MINUTES")
print("   → Lower is better")
# If RMSE >> MAE, you have some very bad predictions")

# Display results
print("\n")
print("RESULTS COMPARISON")

print("\n HUBER REGRESSION:")
print(f"   R² Score:  {r2_huber:.4f}  →  Explains {r2_huber*100:.2f}% of variance")
print(f"   MAE:       {mae_huber:.2f}")
print(f"   RMSE:      {rmse_huber:.2f}")

print("\n RIDGE REGRESSION:")
print(f"   R² Score:  {r2_ridge:.4f}  →  Explains {r2_ridge*100:.2f}% of variance")
print(f"   MAE:       {mae_ridge:.2f} ")
print(f"   RMSE:      {rmse_ridge:.2f}")

# Compare R² scores
if r2_huber > r2_ridge:
    diff_r2 = (r2_huber - r2_ridge) * 100
    print(f"  • R² is {diff_r2:.2f}% better")
    print(f"  • Explains {diff_r2:.2f}% more of the variance in accident duration")
elif r2_ridge > r2_huber:
    diff_r2 = (r2_ridge - r2_huber) * 100
    print(f"  • R² is {diff_r2:.2f}% better")
    print(f"  • Explains {diff_r2:.2f}% more of the variance in accident duration")
else:
    print()

# Compare MAE
print(f"\n📊 Average Prediction Error (MAE):")
if mae_huber < mae_ridge:
    diff_mae = mae_ridge - mae_huber
    print(f"  • Huber is {diff_mae:.2f} minutes MORE accurate")
elif mae_ridge < mae_huber:
    diff_mae = mae_huber - mae_ridge
    print(f"  • Ridge is {diff_mae:.2f} minutes MORE accurate")
else:
    print(f"  • Both have the same average error")

# STEP 8: ANALYZE FEATURE IMPORTANCE

print("WHICH FEATURES MATTER MOST?")

print("\nWhat are coefficients?")
print("\n")
print("Coefficients tell us how much each feature affects the prediction:")
print("  • POSITIVE coefficient → Feature INCREASES duration")
print("  • NEGATIVE coefficient → Feature DECREASES duration")
print("  • LARGER absolute value → Feature is MORE important")

# Create comparison dataframe
feature_comparison = pd.DataFrame({
    'Feature': features,
    'Huber_Coef': huber.coef_,
    'Ridge_Coef': ridge.coef_,
    'Huber_Abs': np.abs(huber.coef_),
    'Ridge_Abs': np.abs(ridge.coef_),
    'Difference': np.abs(huber.coef_ - ridge.coef_)
})

# Sort by Huber importance
feature_comparison_huber = feature_comparison.sort_values('Huber_Abs', ascending=False)

print("\nHUBER - Feature Importance (sorted):")
for idx, row in feature_comparison_huber.iterrows():
    direction = "↑ INCREASES" if row['Huber_Coef'] > 0 else "↓ DECREASES"
    print(f"{row['Feature']:20s} | {direction:12s} | Impact: {row['Huber_Abs']:.4f}")

# Sort by Ridge importance
feature_comparison_ridge = feature_comparison.sort_values('Ridge_Abs', ascending=False)

print("\nRIDGE - Feature Importance (sorted):")
for idx, row in feature_comparison_ridge.iterrows():
    direction = "↑ INCREASES" if row['Ridge_Coef'] > 0 else "↓ DECREASES"
    print(f"{row['Feature']:20s} | {direction:12s} | Impact: {row['Ridge_Abs']:.4f}")

# Features where models disagree
feature_comparison_diff = feature_comparison.sort_values('Difference', ascending=False)

print("\nFeatures where models DISAGREE most:")

print(feature_comparison_diff.head(5).to_string(index=False))
# STEP 9: VISUALICE COMPARISONS
print("CREATING VISUALIZATIONS")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Huber vs Ridge Regression - Complete Comparison', 
             fontsize=16, fontweight='bold')

# PLOT 1: Actual vs Predicted - HUBER
axes[0, 0].scatter(y_test, y_pred_huber, alpha=0.5, edgecolors='k', linewidth=0.5, c='blue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction Line')
axes[0, 0].set_xlabel('Actual Duration (minutes)', fontsize=11)
axes[0, 0].set_ylabel('Predicted Duration (minutes)', fontsize=11)
axes[0, 0].set_title(f'Huber Regression\nR²={r2_huber:.4f}, MAE={mae_huber:.2f}', 
                     fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# PLOT 2: Actual vs Predicted - RIDGE
axes[0, 1].scatter(y_test, y_pred_ridge, alpha=0.5, edgecolors='k', linewidth=0.5, c='green')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction Line')
axes[0, 1].set_xlabel('Actual Duration (minutes)', fontsize=11)
axes[0, 1].set_ylabel('Predicted Duration (minutes)', fontsize=11)
axes[0, 1].set_title(f'Ridge Regression\nR²={r2_ridge:.4f}, MAE={mae_ridge:.2f}', 
                     fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# PLOT 3: Performance Metrics Comparison
metrics = ['R² Score', 'MAE (min)', 'RMSE (min)']
huber_metrics = [r2_huber, mae_huber, rmse_huber]
ridge_metrics = [r2_ridge, mae_ridge, rmse_ridge]

x_pos = np.arange(len(metrics))
width = 0.35

axes[0, 2].bar(x_pos - width/2, huber_metrics, width, label='Huber', color='blue', alpha=0.7)
axes[0, 2].bar(x_pos + width/2, ridge_metrics, width, label='Ridge', color='green', alpha=0.7)
axes[0, 2].set_xlabel('Metrics', fontsize=11)
axes[0, 2].set_ylabel('Value', fontsize=11)
axes[0, 2].set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
axes[0, 2].set_xticks(x_pos)
axes[0, 2].set_xticklabels(metrics)
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3, axis='y')

# PLOT 4: Residuals - HUBER
residuals_huber = y_test - y_pred_huber
axes[1, 0].scatter(y_pred_huber, residuals_huber, alpha=0.5, edgecolors='k', 
                   linewidth=0.5, c='blue')
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
axes[1, 0].set_xlabel('Predicted Duration (minutes)', fontsize=11)
axes[1, 0].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
axes[1, 0].set_title('Huber Residual Plot', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# PLOT 5: Residuals - RIDGE
residuals_ridge = y_test - y_pred_ridge
axes[1, 1].scatter(y_pred_ridge, residuals_ridge, alpha=0.5, edgecolors='k', 
                   linewidth=0.5, c='green')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
axes[1, 1].set_xlabel('Predicted Duration (minutes)', fontsize=11)
axes[1, 1].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
axes[1, 1].set_title('Ridge Residual Plot', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# PLOT 6: Feature Coefficients Comparison
top_n = 8  # Show top 8 features
top_features = feature_comparison_huber.head(top_n)

x_pos = np.arange(len(top_features))
width = 0.35

axes[1, 2].barh(x_pos - width/2, top_features['Huber_Coef'], width, 
                label='Huber', color='blue', alpha=0.7)
axes[1, 2].barh(x_pos + width/2, top_features['Ridge_Coef'], width, 
                label='Ridge', color='green', alpha=0.7)
axes[1, 2].set_yticks(x_pos)
axes[1, 2].set_yticklabels(top_features['Feature'])
axes[1, 2].set_xlabel('Coefficient Value', fontsize=11)
axes[1, 2].set_title(f'Top {top_n} Feature Coefficients', fontsize=12, fontweight='bold')
axes[1, 2].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print("✓ All visualizations created successfully!")


# STEP 10: OUTLIER ANALYSIS
print("OUTLIER SENSITIVITY ANALYSIS")

print("\nWhat are outliers?")
print("-" * 50)
print("Outliers are data points that are VERY different from others")
print("Example: Most accidents last 30-60 min, but one lasts 500 min → OUTLIER!")
print()
print("We define outliers as predictions with errors > 2 standard deviations")

# Calculate absolute errors
abs_errors_huber = np.abs(residuals_huber)
abs_errors_ridge = np.abs(residuals_ridge)

# Identify outliers (errors > 2 standard deviations)
threshold = 2 * abs_errors_huber.std()
outliers_mask = abs_errors_huber > threshold

num_outliers = outliers_mask.sum()

print(f"\nOutlier Statistics:")
print(f"  • Total test samples: {len(y_test)}")
print(f"  • Outliers found: {num_outliers} ({num_outliers/len(y_test)*100:.1f}%)")
print(f"  • Error threshold: {threshold:.2f} minutes")

if num_outliers > 0:
    print(f"\nPerformance on OUTLIERS:")
    print(f"  • Huber MAE:  {abs_errors_huber[outliers_mask].mean():.2f} min")
    print(f"  • Ridge MAE:  {abs_errors_ridge[outliers_mask].mean():.2f} min")
    
    if abs_errors_huber[outliers_mask].mean() < abs_errors_ridge[outliers_mask].mean():
        improvement = abs_errors_ridge[outliers_mask].mean() - abs_errors_huber[outliers_mask].mean()
        print(f"\n  Huber handles outliers {improvement:.2f} min BETTER!")
        print(f"    This proves Huber is more ROBUST to extreme values!")
    else:
        improvement = abs_errors_huber[outliers_mask].mean() - abs_errors_ridge[outliers_mask].mean()
        print(f"\n  ✓ Ridge handles outliers {improvement:.2f} min BETTER!")
        print(f"    (Unusual! Ridge not designed for outliers)")
    
    print(f"\nPerformance on NORMAL CASES (non-outliers):")
    print(f"  • Huber MAE:  {abs_errors_huber[~outliers_mask].mean():.2f} min")
    print(f"  • Ridge MAE:  {abs_errors_ridge[~outliers_mask].mean():.2f} min")
else:
    print("\nNo significant outliers detected in predictions")
    
# STEP 11: FINAL RECOMMENDATIONS
print("FINAL SUMMARY & RECOMMENDATIONS")

print("\nRESULTS FOR YOUR TRAFFIC ACCIDENT DATA:")

# Determine best model
if r2_huber > r2_ridge:
    winner = "HUBER"
    diff = (r2_huber - r2_ridge) * 100
    print(f"ECOMMENDED MODEL: HUBER REGRESSION")
    print(f"   • {diff:.2f}% better R² score")
    print(f"   • Better at handling outliers in your data")
    print(f"   • Your data likely contains extreme accident durations")
elif r2_ridge > r2_huber:
    winner = "RIDGE"
    diff = (r2_ridge - r2_huber) * 100
    print(f"RECOMMENDED MODEL: RIDGE REGRESSION")
    print(f"   • {diff:.2f}% better R² score")
    print(f"   • Better at handling feature correlations")
    print(f"   • Your features might be multicollinear")
else:
    winner = "TIE"
    print(f" BOTH MODELS PERFORM EQUALLY")
    print(f"   • Choose Huber if you suspect outliers")
    print(f"   • Choose Ridge if you suspect multicollinearity")

print(f"\n NEXT STEPS TO IMPROVE:")
print("  1. Tune hyperparameters (epsilon for Huber, alpha for Ridge)")
print("  2. Check for multicollinearity (correlation matrix)")
print("  3. Try feature engineering (interaction terms, polynomial features)")
print("  4. Consider ensemble methods (combining multiple models)")
print("  5. Analyze residuals to find patterns in prediction errors")

print("ANALYSIS COMPLETE!")

print(f"\nBest Model for Your Data: {winner}")
print(f"Best R² Score: {max(r2_huber, r2_ridge):.4f}")
print(f"Best MAE: {min(mae_huber, mae_ridge):.2f} minutes")
print("\nYou can now use these insights to make better predictions!")
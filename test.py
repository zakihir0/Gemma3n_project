from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
import sys

# --- Data Preprocessing ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_final)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_train_final.values, test_size=0.2, random_state=42
)

# --- Build the Model (Multi-output) ---
def build_multi_output_model(input_shape, output_shape):
    """Builds a neural network for multi-output regression."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=[input_shape]),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        # The output layer must have one neuron for each target wavelength
        tf.keras.layers.Dense(output_shape)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_model(input_shape, output_shape):
    """Builds a CNN-based neural network for multi-output regression."""
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_shape, 1), input_shape=[input_shape]),
        tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        # The output layer must have one neuron for each target wavelength
        tf.keras.layers.Dense(output_shape)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_resnet_model(input_shape, output_shape):
    """Builds a ResNet-inspired neural network for multi-output regression."""
    inputs = tf.keras.layers.Input(shape=[input_shape])
    
    # First block
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Residual blocks
    for _ in range(3):
        residual = x
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, residual])
    
    # Output layers
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    # The output layer must have one neuron for each target wavelength
    outputs = tf.keras.layers.Dense(output_shape)(x)
    
    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_attention_model(input_shape, output_shape):
    """Builds an attention-based neural network for multi-output regression."""
    inputs = tf.keras.layers.Input(shape=[input_shape])
    
    # Transform to sequence-like format
    x = tf.keras.layers.Reshape((input_shape, 1))(inputs)
    
    # Multi-head attention
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=8, key_dim=64
    )(x, x)
    
    # Add & Norm
    x = tf.keras.layers.Add()([x, attention_output])
    x = tf.keras.layers.LayerNormalization()(x)
    
    # Feed forward
    ff_output = tf.keras.layers.Dense(256, activation='relu')(x)
    ff_output = tf.keras.layers.Dense(1)(ff_output)
    
    # Add & Norm
    x = tf.keras.layers.Add()([x, ff_output])
    x = tf.keras.layers.LayerNormalization()(x)
    
    # Global pooling and output
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    # The output layer must have one neuron for each target wavelength
    outputs = tf.keras.layers.Dense(output_shape)(x)
    
    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# --- Create and Train All Models ---
input_shape = X_train.shape[1]
output_shape = y_train.shape[1] # Number of target wavelengths

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Model configurations
model_configs = [
    ("Neural Network", build_multi_output_model, 100),
    ("CNN", build_cnn_model, 50),
    ("ResNet", build_resnet_model, 50),
    ("Attention", build_attention_model, 50)
]

all_models = {}
training_histories = {}

# Create and train all models
for model_name, build_func, epochs in model_configs:
    print(f"\n{'='*50}")
    print(f"Creating {model_name} model...")
    print(f"{'='*50}")
    
    model_instance = build_func(input_shape, output_shape)
    model_instance.summary()
    
    print(f"\nTraining {model_name} model...")
    history = model_instance.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[early_stopping]
    )
    
    all_models[model_name] = model_instance
    training_histories[model_name] = history

print("\nAll neural network models trained successfully!")

# Use the original Neural Network model for submission
model = all_models["Neural Network"]


# --- Process the Test Set ---
# Get a list of test planet IDs from the sample submission file
test_planet_ids = sample_submission_df['planet_id'].unique()

all_test_features = []
corresponding_test_ids = []

for planet_id in tqdm(test_planet_ids, desc="Creating Test Features"):
    features = load_and_process_planet_data(planet_id, TEST_DIR)
    if features is not None:
        all_test_features.append(features)
        corresponding_test_ids.append(planet_id)

# Create the final test feawture matrix
X_test_processed = pd.DataFrame(all_test_features)

# Scale the test features using the *same scaler* fitted on the training data
X_test_scaled = scaler.transform(X_test_processed)

# --- Generate Predictions ---
print("Generating test predictions...")
predictions = model.predict(X_test_scaled)

# --- Create Submission File ---
# Create a DataFrame with the predictions
pred_df = pd.DataFrame(predictions, columns=y_train_final.columns)
pred_df['planet_id'] = corresponding_test_ids

# Reorder columns to match submission format if necessary
pred_df = pred_df[['planet_id'] + list(y_train_final.columns)]

# Save to csv
pred_df.to_csv('submission.csv', index=False)

print("\nSubmission file created successfully!")
print(pred_df.head())



# --- Model Performance Evaluation ---
def evaluate_model(model, X_val, y_val, model_name):
    """Evaluate model performance and return metrics."""
    predictions = model.predict(X_val)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, predictions)
    
    print(f"\n{model_name} Performance:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²: {r2:.6f}")
    
    return {'mse': mse, 'rmse': rmse, 'r2': r2, 'predictions': predictions}

# Evaluate all models
print("Evaluating model performance on validation set...")

results_summary = {}
for model_name, model_instance in all_models.items():
    results_summary[model_name] = evaluate_model(model_instance, X_val, y_val, model_name)

print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
for model_name, results in results_summary.items():
    print(f"{model_name:20} | RMSE: {results['rmse']:.6f} | R²: {results['r2']:.6f}")



# --- Visualization and Analysis ---
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# 1. RMSE Comparison
models = list(results_summary.keys())
rmse_values = [results_summary[model]['rmse'] for model in models]
r2_values = [results_summary[model]['r2'] for model in models]

axes[0, 0].bar(models, rmse_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
axes[0, 0].set_title('RMSE Comparison')
axes[0, 0].set_ylabel('RMSE')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. R² Comparison
axes[0, 1].bar(models, r2_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
axes[0, 1].set_title('R² Score Comparison')
axes[0, 1].set_ylabel('R² Score')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Prediction vs Actual (best model)
best_model_name = min(results_summary.keys(), key=lambda x: results_summary[x]['rmse'])
best_predictions = results_summary[best_model_name]['predictions']

# Flatten for plotting (taking first wavelength as example)
y_true_flat = y_val[:, 0] if len(y_val.shape) > 1 else y_val
y_pred_flat = best_predictions[:, 0] if len(best_predictions.shape) > 1 else best_predictions

axes[1, 0].scatter(y_true_flat, y_pred_flat, alpha=0.6, color='#FF6B6B')
axes[1, 0].plot([y_true_flat.min(), y_true_flat.max()], [y_true_flat.min(), y_true_flat.max()], 'k--', lw=2)
axes[1, 0].set_xlabel('Actual Values')
axes[1, 0].set_ylabel('Predicted Values')
axes[1, 0].set_title(f'Predictions vs Actual ({best_model_name})')

# 4. Residuals plot
residuals = y_true_flat - y_pred_flat
axes[1, 1].scatter(y_pred_flat, residuals, alpha=0.6, color='#4ECDC4')
axes[1, 1].axhline(y=0, color='k', linestyle='--')
axes[1, 1].set_xlabel('Predicted Values')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title(f'Residuals Plot ({best_model_name})')

plt.tight_layout()
plt.show()

print(f"\nBest performing model: {best_model_name}")
print(f"RMSE: {results_summary[best_model_name]['rmse']:.6f}")
print(f"R²: {results_summary[best_model_name]['r2']:.6f}")
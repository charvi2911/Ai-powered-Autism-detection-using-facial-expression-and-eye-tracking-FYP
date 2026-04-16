from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from PIL import Image
import os
import io
from werkzeug.utils import secure_filename
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib.pyplot as plt
import base64
import joblib
import pickle
# Database connection
mydb = mysql.connector.connect(
    host='localhost',
    port=3306,          
    user='root',        
    passwd='',          
    database='autism_detection'
)

mycur = mydb.cursor()

def create_tables_if_not_exist():
    create_predictions_table = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        prediction_result VARCHAR(50) NOT NULL,
        confidence FLOAT NOT NULL,
        image_path VARCHAR(500),
        prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    create_users_table = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL,
        address TEXT,
        registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    mycur.execute(create_users_table)
    mycur.execute(create_predictions_table)
    mydb.commit()
    print("Tables created/verified successfully")

create_tables_if_not_exist()

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_autism_detection_application'


# -----------------------------------------
# 1. LOAD & PREPROCESS DATA
# -----------------------------------------
metadata_df = pd.read_csv('Metadata_Participants.csv')
dataset_df = pd.read_csv('11.csv')

merged_df_new = pd.merge(metadata_df, dataset_df, left_on='ParticipantID', right_on='Unnamed: 0', how='inner')

selected_columns = [
    'ParticipantID', 'Gender', 'Age', 'Class',
    'Category Right', 'Category Left', 'Index Right', 'Index Left',
    'Pupil Size Right X [px]', 'Pupil Size Right Y [px]', 'Pupil Diameter Right [mm]',
    'Pupil Size Left X [px]', 'Pupil Size Left Y [px]', 'Pupil Diameter Left [mm]',
    'Point of Regard Right X [px]', 'Point of Regard Right Y [px]',
    'Point of Regard Left X [px]', 'Point of Regard Left Y [px]',
    'Gaze Vector Right X', 'Gaze Vector Right Y', 'Gaze Vector Right Z',
    'Gaze Vector Left X', 'Gaze Vector Left Y', 'Gaze Vector Left Z',
    'Eye Position Right X [mm]', 'Eye Position Right Y [mm]', 'Eye Position Right Z [mm]',
    'Eye Position Left X [mm]', 'Eye Position Left Y [mm]', 'Eye Position Left Z [mm]',
    'Pupil Position Right X [px]', 'Pupil Position Right Y [px]',
    'Pupil Position Left X [px]', 'Pupil Position Left Y [px]'
]

selected_df = merged_df_new[selected_columns]

# Remove first 7 rows
selected_df_cleaned = selected_df.drop(index=selected_df.index[:7])

# Convert numeric columns
columns_to_convert = [
    'Index Right','Index Left','Pupil Size Right X [px]', 'Pupil Size Right Y [px]', 'Pupil Diameter Right [mm]',
    'Pupil Size Left X [px]', 'Pupil Size Left Y [px]', 'Pupil Diameter Left [mm]',
    'Point of Regard Right X [px]', 'Point of Regard Right Y [px]',
    'Point of Regard Left X [px]', 'Point of Regard Left Y [px]',
    'Gaze Vector Right X', 'Gaze Vector Right Y', 'Gaze Vector Right Z',
    'Gaze Vector Left X', 'Gaze Vector Left Y', 'Gaze Vector Left Z',
    'Eye Position Right X [mm]', 'Eye Position Right Y [mm]', 'Eye Position Right Z [mm]',
    'Eye Position Left X [mm]', 'Eye Position Left Y [mm]', 'Eye Position Left Z [mm]',
    'Pupil Position Right X [px]', 'Pupil Position Right Y [px]', 
    'Pupil Position Left X [px]', 'Pupil Position Left Y [px]'
]

selected_df_cleaned[columns_to_convert] = selected_df_cleaned[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Encode categorical columns
label_encoder = LabelEncoder()
selected_df_cleaned['Gender'] = label_encoder.fit_transform(selected_df_cleaned['Gender'])
selected_df_cleaned['Class'] = label_encoder.fit_transform(selected_df_cleaned['Class'])
selected_df_cleaned['Category Right'] = label_encoder.fit_transform(selected_df_cleaned['Category Right'])
selected_df_cleaned['Category Left'] = label_encoder.fit_transform(selected_df_cleaned['Category Left'])

# -----------------------------------------
# 2. BOOTSTRAP AUGMENTATION
# -----------------------------------------
n_needed = 1000 - len(selected_df_cleaned)
bootstrap_df = selected_df_cleaned.sample(n=n_needed, replace=True, random_state=42)
augmented_df = pd.concat([selected_df_cleaned, bootstrap_df], ignore_index=True)

# -----------------------------------------
# 3. FEATURE SELECTION (TOP 20)
# -----------------------------------------
X = augmented_df.drop(columns=['ParticipantID', 'Class'])
Y = augmented_df['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

feature_importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

top_n_features = feature_importance_df.head(20)['Feature'].tolist()
#  top_n_features = feature_importance_df.head(20)['Feature'].tolist() ---

print("\n" + "="*50)
print("PRIMARY EYE-TRACKING PREDICTORS (TOP 5)")
print("="*50)

# We extract the top 5 for the display
top_5_analysis = feature_importance_df.head(5)

for i, (index, row) in enumerate(top_5_analysis.iterrows()):
    feature_name = row['Feature']
    importance_score = row['Importance']
    print(f"RANK {i+1}: {feature_name:<30} | {importance_score:.4f}")

print("="*50)
print("NOTE: These features contribute the most to the model decision.")
print("="*50 + "\n")
# Filter only top-20
X_train_selected = X_train[top_n_features]
X_test_selected = X_test[top_n_features]

# -----------------------------------------
# 4. STANDARD SCALING
# -----------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# -----------------------------------------
# 5. TRAIN SVM
# -----------------------------------------
svm_model = SVC(probability=True)
svm_model.fit(X_train_scaled, Y_train)

# --- ADD THIS IMMEDIATELY AFTER svm_model.fit(X_train_scaled, Y_train) ---

import joblib
import pickle

# Save the models so the web app can use them
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Save the feature names so we know which 20 to use during prediction
with open('feature_info.pkl', 'wb') as f:
    pickle.dump(top_n_features, f)

print("All models and feature info saved successfully for expo!")

print("Model trained successfully!")
# Load the SVM model and preprocessing objects for eye-tracking prediction
# Load pre-trained models
svm_model = joblib.load('svm_model.pkl')
scaler_eyetracking = joblib.load('scaler.pkl')
label_encoder_eyetracking = joblib.load('label_encoder.pkl')
with open('feature_info.pkl', 'rb') as f:
    feature_info_eyetracking = pickle.load(f)

print("Eye-tracking models and preprocessing objects loaded successfully")
# Configuration
UPLOAD_FOLDER = 'static/uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

photo_size = 224

def create_mobilenet_model():
    base_model = MobileNet(input_shape=(photo_size, photo_size, 3), include_top=False, weights='imagenet')
    
    for layer in base_model.layers:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(base_model.input, output)
    model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

mobilenet_model = tf.keras.models.load_model("mobilenet_model.h5")
print("MobileNet model loaded successfully from file")

model_performance = {
    'MobileNet': {
        'Accuracy': '81.67%',
        'Precision': '79.14%', 
        'Recall': '86.00%',
        'F1-Score': '82.43%',
        'Specificity': '77.33%'
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img = img.resize((photo_size, photo_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Use the same preprocessing as during training: rescale=1/255
    img_array = img_array / 255.0
    
    print(f"Preprocessed image - Min: {np.min(img_array):.4f}, Max: {np.max(img_array):.4f}")
    
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()
        confirmpassword = request.form.get('confirmpassword', '').strip()
        address = request.form.get('address', '').strip()

        if not all([name, email, password, confirmpassword, address]):
            flash('All fields are required!', 'danger')
            return render_template('registration.html')

        if password != confirmpassword:
            flash('Passwords do not match!', 'danger')
            return render_template('registration.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters long!', 'danger')
            return render_template('registration.html')

        sql = 'SELECT * FROM users WHERE email = %s'
        val = (email,)
        mycur.execute(sql, val)
        data = mycur.fetchone()

        if data:
            flash('User already registered!', 'danger')
            return render_template('registration.html')
        else:
            hashed_password = generate_password_hash(password)
            sql = 'INSERT INTO users (name, email, password, Address) VALUES (%s, %s, %s, %s)'
            val = (name, email, hashed_password, address)
            mycur.execute(sql, val)
            mydb.commit()
            flash('User registered successfully! Please login.', 'success')
            return redirect(url_for('login'))

    return render_template('registration.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()

        if not email or not password:
            flash('Please enter both email and password!', 'danger')
            return render_template('login.html')

        sql = 'SELECT id, name, email, password, Address FROM users WHERE email = %s'
        val = (email,)
        mycur.execute(sql, val)
        data = mycur.fetchone()

        if data:
            stored_password = data[3]
            if check_password_hash(stored_password, password):
                session['user_id'] = data[0]
                session['user_email'] = data[2]
                session['user_name'] = data[1]
                flash('Login successful!', 'success')
                return redirect(url_for('home'))
            else:
                flash('Invalid email or password!', 'danger')
                return render_template('login.html')
        else:
            flash('User with this email does not exist. Please register.', 'danger')
            return redirect(url_for('registration'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', username=session.get('user_name'))

@app.route('/about')
def about():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('about.html')

@app.route('/models')
def models():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('models.html', model_performance=model_performance)



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))

    prediction_result = None
    image_path = None
    filename = None
    raw_prediction = None 
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No image selected for upload', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img = Image.open(filepath)
            processed_image = preprocess_image(img)
            
            if processed_image is not None and mobilenet_model is not None:
                prediction = mobilenet_model.predict(processed_image, verbose=0)
                raw_prediction = float(prediction[0][0])
                print(f"Raw prediction value: {raw_prediction:.6f}")
                
                # Direct prediction based on raw value only
                if raw_prediction < 0.5:
                    prediction_result = 'Autistic Traits Detected'
                else:
                    prediction_result = 'No Autistic Traits Detected'
                
                print(f"Prediction: {prediction_result}, Raw Value: {raw_prediction:.6f}")
                
                # Modified SQL insert without confidence
                sql = '''
                    INSERT INTO predictions 
                    (user_id, prediction_result, image_path) 
                    VALUES (%s, %s, %s)
                '''
                val = (
                    session['user_id'], 
                    prediction_result, 
                    filepath
                )
                mycur.execute(sql, val)
                mydb.commit()
                
                flash('Prediction completed successfully!', 'success')
                image_path = filepath
            else:
                flash('Error processing image or model not available', 'danger')
        else:
            flash('Allowed image types are: png, jpg, jpeg, gif', 'danger')
    
    return render_template('predict.html', 
                         prediction_result=prediction_result,
                         image_path=image_path,
                         filename=filename,
                         raw_prediction=raw_prediction)

@app.route("/predict2", methods=["GET", "POST"])
def predict2():
    result = ""
    if request.method == "POST":
        try:
            user_input = []
            for col in top_n_features:
                val = float(request.form[col])
                user_input.append(val)

            user_array = np.array(user_input).reshape(1, -1)
            user_scaled = scaler.transform(user_array)

            pred = svm_model.predict(user_scaled)[0]

            if pred == 0:
                result = "Prediction: ASD (0)"
            else:
                result = "Prediction: TD (1)"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('predict2.html', features=top_n_features, result=result)




@app.route('/debug_prediction', methods=['POST'])
def debug_prediction():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"debug_{filename}")
    file.save(filepath)
    
    img = Image.open(filepath)
    processed_image = preprocess_image(img)
    
    if processed_image is not None and mobilenet_model is not None:
        prediction = mobilenet_model.predict(processed_image, verbose=0)
        raw_value = float(prediction[0][0])
        
        return jsonify({
            'raw_prediction': raw_value,
            'interpretation_standard': 'Autistic' if raw_value < 0.5 else 'Non-Autistic',
            'interpretation_reversed': 'Autistic' if raw_value > 0.5 else 'Non-Autistic',
        })
    else:
        return jsonify({'error': 'Model or image processing failed'}), 500

@app.route('/test_thresholds', methods=['POST'])
def test_thresholds():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"test_{filename}")
    file.save(filepath)
    
    img = Image.open(filepath)
    processed_image = preprocess_image(img)
    
    if processed_image is not None and mobilenet_model is not None:
        prediction = mobilenet_model.predict(processed_image, verbose=0)
        raw_value = float(prediction[0][0])
        
        results = []
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for threshold in thresholds:
            if raw_value < threshold:
                result_standard = "Autistic"
                conf_standard = (1 - raw_value) * 100
            else:
                result_standard = "Non-Autistic"
                conf_standard = raw_value * 100
                
            if raw_value > threshold:
                result_reversed = "Autistic"
                conf_reversed = raw_value * 100
            else:
                result_reversed = "Non-Autistic"
                conf_reversed = (1 - raw_value) * 100
                
            results.append({
                'threshold': threshold,
                'standard': f"{result_standard} ({conf_standard:.1f}%)",
                'reversed': f"{result_reversed} ({conf_reversed:.1f}%)"
            })
        
        return jsonify({
            'raw_prediction': raw_value,
            'results': results
        })
    else:
        return jsonify({'error': 'Model or image processing failed'}), 500

@app.route('/model_info')
def model_info():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    stream = io.StringIO()
    mobilenet_model.summary(print_fn=lambda x: stream.write(x + '\n'))
    model_summary = stream.getvalue()
    stream.close()
    
    training_info = {
        'input_size': f'{photo_size}x{photo_size}',
        'preprocessing': 'rescale=1/255',
        'classes': ['autistic (0)', 'non_autistic (1)'],
        'architecture': 'MobileNet + GlobalAveragePooling2D + Dense(128) + Dropout(0.2) + Dense(1)',
        'activation': 'sigmoid',
        'optimizer': 'RMSprop(learning_rate=0.0001)',
        'loss': 'binary_crossentropy'
    }
    
    return render_template('model_info.html', 
                         model_summary=model_summary,
                         training_info=training_info)

if __name__ == '__main__':
    app.run(debug=True)
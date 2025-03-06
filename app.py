from flask import Flask, render_template, redirect, url_for, request, flash, send_from_directory, jsonify
from flask import Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_migrate import Migrate
from forms import LoginForm, RegistrationForm, RegistrationForm1, RegistrationForm2, ReportForm, ChatForm, MessageForm
import datetime
import os
import fitz
from twilio.rest import Client
from phonenumbers import parse, is_valid_number
from sqlalchemy import or_, and_
import PyPDF2
from transformers import pipeline
import cv2
from tensorflow import keras
from keras._tf_keras.keras.models import load_model  # Correct import statement
from pdf2image import convert_from_path
import numpy as np
from PIL import Image
from PyPDF2 import PdfReader
from threading import Thread
from smart_band_simulation import start_simulation, stop_simulation
import smart_band_simulation

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads', 'reports')
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
TWILIO_SID = ''
TWILIO_AUTH_TOKEN = ''
TWILIO_PHONE_NUMBER = ''

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

def send_sms(to_number, message):
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=to_number
    )

# Create the upload folder if it does not exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)
migrate = Migrate(app, db)  # Initialize Flask-Migrate

login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    phone_number = db.Column(db.String(20), nullable=True)  # Add this line
    reports = db.relationship('Report', backref='user', lazy=True)

class Staff(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    

class Doctor(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    phone_number = db.Column(db.String(20), nullable=True)  # Add this line
    reports = db.relationship('Report', backref='doctor', lazy=True)

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=False)
    report_type = db.Column(db.String(50), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    summary = db.Column(db.Text, nullable=True)  # Add this line
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    approved = db.Column(db.Boolean, default=False, nullable=False)  # Track approval status

class SmartBandData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)
    blood_pressure = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, nullable=False)
    receiver_id = db.Column(db.Integer, nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'), nullable=False)
    messages = db.relationship('Message', backref='conversation', lazy=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, nullable=False)  # ID of the sender (either patient or doctor)
    receiver_id = db.Column(db.Integer, nullable=False)  # ID of the receiver (either doctor or patient)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def home():
    return render_template('home.html', title='Home')

@app.route('/Hospital')
def Hospital():
    return render_template('Hospital.html', title='Hospital')

@app.route('/patient')
@login_required
def patient():
    return render_template('patient.html', title='Patient')


@app.route('/uploads/<filename>')
def get_report(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

from sqlalchemy import or_

@app.route('/start_simulation', methods=['POST'])
@login_required
def start_simulation_route():
    """Handle start simulation request."""
    user_id = current_user.id
    smart_band_simulation.start_simulation(user_id)
    return Response(status=204)

@app.route('/stop_simulation', methods=['POST'])
@login_required
def stop_simulation_route():
    """Handle stop simulation request."""
    smart_band_simulation.stop_simulation()
    return Response(status=204)


@app.route('/chat/<int:receiver_id>', methods=['GET', 'POST'])
@login_required
def chat(receiver_id):
    # Determine if the receiver is a doctor or patient
    if current_user.__class__.__name__ == 'User':  # Patient
        receiver = Doctor.query.get_or_404(receiver_id)
    else:  # Doctor
        receiver = User.query.get_or_404(receiver_id)

    # Find or create a conversation
    conversation = Conversation.query.filter(
        or_(
            and_(
                Conversation.patient_id == current_user.id,
                Conversation.doctor_id == receiver_id
            ),
            and_(
                Conversation.patient_id == receiver_id,
                Conversation.doctor_id == current_user.id
            )
        )
    ).first()

    if not conversation:
        conversation = Conversation(patient_id=current_user.id, doctor_id=receiver_id)
        db.session.add(conversation)
        db.session.commit()

    if request.method == 'POST':
        content = request.form.get('message')
        if content:
            message = Message(
                sender_id=current_user.id,
                receiver_id=receiver_id,
                content=content,
                conversation_id=conversation.id
            )
            db.session.add(message)
            db.session.commit()
            flash('Message sent!', 'success')

    messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.timestamp.asc()).all()
    return render_template('chat.html', conversation=conversation, messages=messages, receiver=receiver)


@app.route('/doctor', methods=['GET', 'POST'])
def doctor():
    form = LoginForm()
    if form.validate_on_submit():
        doctor = Doctor.query.filter_by(username=form.username.data).first()
        if doctor and check_password_hash(doctor.password, form.password.data):
            login_user(doctor, remember=form.remember.data)
            return redirect(url_for('doctordash'))
        else:
            flash('Login failed. Incorrect username or password.', 'danger')
    return render_template('doctor.html', form=form, title='Doctor Login')

@app.route('/doctordash', methods=['GET', 'POST'])
@login_required
def doctordash():
    reports = Report.query.filter_by(doctor_id=current_user.id, approved=False).all()

    if request.method == 'POST':
        report_id = request.form.get('report_id')
        report = Report.query.get(report_id)
        report.approved = True
        db.session.commit()

        # Find the user based on patient_name (which should match the username)
        patient = User.query.filter_by(username=report.patient_name).first()

        if patient:
            # Send SMS to the patient after approval
            sms_message = f"Dear {patient.username}, Your report titled '{report.report_type}' has been reviewed and is available on your dashboard. Summary:\n{report.summary}"
            send_sms(patient.phone_number, sms_message)

            flash('Report approved and SMS sent to the patient.', 'success')
        else:
            flash('Patient not found, SMS not sent.', 'danger')

    # After approval, the report should remain in the dashboard, but only the SMS part should be handled
    reports = Report.query.filter_by(doctor_id=current_user.id).all()

    return render_template('doctordash.html', title='Doctor Dashboard', reports=reports)

   
@app.route('/staff', methods=['GET', 'POST'])
def staff():
    form = LoginForm()
    if form.validate_on_submit():
        staff = Staff.query.filter_by(username=form.username.data).first()
        if staff and check_password_hash(staff.password, form.password.data):
            login_user(staff, remember=form.remember.data)
            return redirect(url_for('upload_report'))
        else:
            flash('Login failed. Incorrect password.', 'danger')
    return render_template('Staff.html', title='Staff', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        user = User(
            username=form.username.data,
            email=form.email.data,
            phone_number=form.phone_number.data,  # Capture phone number
            password=hashed_password
        )
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/Staffreg', methods=['GET', 'POST'])
def Staffreg():
    form = RegistrationForm1()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        staff = Staff(username=form.username.data, password=hashed_password)
        db.session.add(staff)
        db.session.commit()
        flash('Your account has been created!', 'success')
        return redirect(url_for('Hospital'))
    return render_template('Staffreg.html', title='Staff Register', form=form)

@app.route('/Doctorreg', methods=['GET', 'POST'])
def Doctorreg():
    form = RegistrationForm2()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        doctor = Doctor(username=form.username.data, password=hashed_password, phone_number=form.phone_number.data)
        db.session.add(doctor)
        db.session.commit()
        flash('Your account has been created!', 'success')
        return redirect(url_for('Hospital'))
    return render_template('Doctorreg.html', title='Doctor Register', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    reports = Report.query.filter_by(user_id=current_user.id).all()
    smart_band_data = SmartBandData.query.filter_by(user_id=current_user.id).order_by(SmartBandData.timestamp.desc()).all()
    return render_template('dashboard.html', title='Dashboard', reports=reports, smart_band_data=smart_band_data)

# Initialize the NLP model for text summarization
nlp_model = pipeline("summarization", model="facebook/bart-large-cnn")

# Load a pre-trained CNN model for ECG image analysis
# Ensure you have trained this model and saved it at the specified path
ecg_model = load_model('fine_tuned_ecg_model.h5')

def extract_pdf_text(filepath):
    with open(filepath, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_images_from_pdf(filepath):
    images = convert_from_path(filepath)
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"page_{i}.png")
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
    return image_paths

def find_ecg_image(image_paths):
    # You can enhance this function with pattern recognition or OCR
    for image_path in image_paths:
        # Simplified: assume the first image is the ECG for now
        return image_path
    return None

def process_ecg_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert grayscale to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize the image
    image_rgb = cv2.resize(image_rgb, (224, 224))
    
    # Normalize the image
    image_rgb = image_rgb / 255.0
    
    # Reshape the image to fit the model input
    image_rgb = image_rgb.reshape(1, 224, 224, 3)
    
    # Predict using the model
    prediction = ecg_model.predict(image_rgb)
    
    if prediction[0][0] > 0.5:
        result = "Abnormal ECG detected."
    else:
        result = "Normal ECG."
    
    return result


def extract_pdf_details(filepath, report_type):
    pdf_text = extract_pdf_text(filepath)
    if report_type.lower() == 'ecg':
        summary = "ECG report processed. No text summarization applied."
        image_paths = extract_images_from_pdf(filepath)
        ecg_result = ""
        ecg_image_path = find_ecg_image(image_paths)
        if ecg_image_path:
            ecg_result = process_ecg_image(ecg_image_path)
        combined_summary = summary + "\n\n" + ecg_result
    elif report_type.lower() == 'blood pressure':
        summary = nlp_model(pdf_text, max_length=150, min_length=30, do_sample=False)
        combined_summary = summary[0]['summary_text']
    else:
        combined_summary = "Unknown report type."

    return combined_summary

@app.route('/upload_report', methods=['GET', 'POST'])
@login_required
def upload_report():
    form = ReportForm()
    form.doctor_id.choices = [(doctor.id, doctor.username) for doctor in Doctor.query.all()]
    form.user_id.choices = [(user.id, user.username) for user in User.query.all()]

    if form.validate_on_submit():
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            relative_filepath = filename
            
            # Determine report type and process accordingly
            report_summary = extract_pdf_details(filepath, form.report_type.data)

            report = Report(
                patient_name=form.patient_name.data,
                report_type=form.report_type.data,
                file_path=relative_filepath,
                summary=report_summary,  # Save the summary
                user_id=form.user_id.data,
                doctor_id=form.doctor_id.data,
            )
            db.session.add(report)
            db.session.commit()

            # Send SMS alert to the doctor
            doctor_phone_number = Doctor.query.get(form.doctor_id.data).phone_number
            sms_message = f"Dear Dr. {Doctor.query.get(form.doctor_id.data).username}, A new report has been uploaded for your review. Summary: \n{report_summary}"
            send_sms(doctor_phone_number, sms_message)

            flash('Report uploaded. The doctor has been alerted to review the report.', 'success')
            return redirect(url_for('upload_report'))
    
    return render_template('upload_report.html', form=form)



@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/api/smart_band_data', methods=['POST'])
def receive_smart_band_data():
    data = request.json
    user_id = data['user_id']
    heart_rate = data['heart_rate']
    blood_pressure = data['blood_pressure']
    timestamp = datetime.datetime.utcnow()
    smart_band_data = SmartBandData(user_id=user_id, heart_rate=heart_rate, blood_pressure=blood_pressure, timestamp=timestamp)
    db.session.add(smart_band_data)
    db.session.commit()
    return jsonify({'message': 'Data received successfully'}), 201

if __name__ == '__main__':
    app.run(debug=True)

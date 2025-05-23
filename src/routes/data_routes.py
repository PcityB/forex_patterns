from flask import Blueprint, render_template, request, jsonify, current_app, flash, redirect, url_for, send_file
import os
import pandas as pd
import json
from werkzeug.utils import secure_filename
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
data_bp = Blueprint('data', __name__, url_prefix='/data')

# Ensure directories exist
def ensure_directories():
    os.makedirs(os.path.join(current_app.config['UPLOAD_FOLDER']), exist_ok=True)
    os.makedirs(os.path.join(current_app.config['PROCESSED_FOLDER']), exist_ok=True)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@data_bp.route('/')
def index():
    """Data management dashboard"""
    # Get list of available raw data files
    raw_files = []
    if os.path.exists(current_app.config['UPLOAD_FOLDER']):
        raw_files = [f for f in os.listdir(current_app.config['UPLOAD_FOLDER']) 
                    if os.path.isfile(os.path.join(current_app.config['UPLOAD_FOLDER'], f)) 
                    and f.endswith('.csv')]
    
    # Get list of available processed data files
    processed_files = []
    if os.path.exists(current_app.config['PROCESSED_FOLDER']):
        processed_files = [f for f in os.listdir(current_app.config['PROCESSED_FOLDER']) 
                          if os.path.isfile(os.path.join(current_app.config['PROCESSED_FOLDER'], f)) 
                          and f.endswith('.csv')]
    
    # Get list of all data files in the data directory
    data_dir = os.path.join(current_app.root_path, '../../forex_pattern_framework/data')
    data_files = []
    if os.path.exists(data_dir):
        data_files = [f for f in os.listdir(data_dir)
                      if os.path.isfile(os.path.join(data_dir, f))]

    return render_template('data/index.html',
                           raw_files=raw_files,
                           processed_files=processed_files,
                           data_files=data_files)

@data_bp.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Upload data file endpoint"""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            ensure_directories()
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            flash(f'File {filename} uploaded successfully', 'success')
            return redirect(url_for('data.index'))
        else:
            flash(f'Invalid file type. Allowed types: {", ".join(current_app.config["ALLOWED_EXTENSIONS"])}', 'danger')
            return redirect(request.url)
    
    return render_template('data/upload.html')

@data_bp.route('/preview/<filename>')
def preview_file(filename):
    """Preview data file contents"""
    try:
        # Determine if it's a raw or processed file
        if os.path.exists(os.path.join(current_app.config['PROCESSED_FOLDER'], filename)):
            file_path = os.path.join(current_app.config['PROCESSED_FOLDER'], filename)
            file_type = 'processed'
        elif os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], filename)):
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file_type = 'raw'
        else:
            flash(f'File {filename} not found', 'danger')
            return redirect(url_for('data.index'))
        
        # Read first few rows of the file
        df = pd.read_csv(file_path, nrows=100)
        
        # Get basic info
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        row_count = len(pd.read_csv(file_path))
        columns = df.columns.tolist()
        
        # Convert to HTML table
        table_html = df.head(20).to_html(classes='table table-striped table-bordered')
        
        return render_template('data/preview.html', 
                              filename=filename,
                              file_size=f"{file_size:.2f} MB",
                              row_count=row_count,
                              columns=columns,
                              table_html=table_html)
    
    except Exception as e:
        flash(f'Error previewing file: {str(e)}', 'danger')
        logger.error(f"Error previewing file: {str(e)}", exc_info=True)
        return redirect(url_for('data.index'))

@data_bp.route('/preprocess', methods=['GET', 'POST'])
def preprocess_data():
    """Preprocess data file"""
    if request.method == 'POST':
        try:
            # Get form data
            filename = request.form.get('filename')
            handle_missing = request.form.get('handle_missing', 'ffill')
            normalize = request.form.get('normalize', 'false') == 'true'
            engineer_features = request.form.get('engineer_features', 'false') == 'true'
            
            # Import preprocessing module
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from data_preprocessing import ForexDataPreprocessor
            
            # Setup paths
            input_file = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            output_dir = current_app.config['PROCESSED_FOLDER']
            ensure_directories()
            
            # Create preprocessor
            preprocessor = ForexDataPreprocessor(os.path.dirname(input_file))
            
            # Always load the file directly
            preprocessor.load_data(filename)
            
            # Clean data
            preprocessor.clean_data()
            
            # Engineer features if requested
            if engineer_features:
                preprocessor.engineer_features()
            
            # Normalize if requested
            if normalize:
                preprocessor.normalize_data()
            
            # Save processed data
            saved_files = preprocessor.save_processed_data(output_dir)
            
            # Create visualizations if requested
            if request.form.get('create_visualizations', 'false') == 'true':
                viz_dir = os.path.join(current_app.config['STATIC_FOLDER'], 'visualizations')
                os.makedirs(viz_dir, exist_ok=True)
                
                for tf, df in preprocessor.processed_data.items():
                    preprocessor.visualize_data(tf, n_samples=1000, output_dir=viz_dir)
            
            flash(f'Data preprocessing completed successfully. Files saved to {output_dir}', 'success')
            return redirect(url_for('data.index'))
            
        except Exception as e:
            flash(f'Error preprocessing data: {str(e)}', 'danger')
            logger.error(f"Preprocessing error: {str(e)}", exc_info=True)
            return redirect(url_for('data.index'))
    
    # GET request - show preprocessing form
    raw_files = []
    if os.path.exists(current_app.config['UPLOAD_FOLDER']):
        raw_files = [f for f in os.listdir(current_app.config['UPLOAD_FOLDER']) 
                    if os.path.isfile(os.path.join(current_app.config['UPLOAD_FOLDER'], f)) 
                    and f.endswith('.csv')]
    
    return render_template('data/preprocess.html', raw_files=raw_files)

@data_bp.route('/download/<filename>')
def download_file(filename):
    """Download processed data file"""
    try:
        # Determine if it's a raw or processed file
        if os.path.exists(os.path.join(current_app.config['PROCESSED_FOLDER'], filename)):
            file_path = os.path.join(current_app.config['PROCESSED_FOLDER'], filename)
            file_type = 'processed'
        elif os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], filename)):
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file_type = 'raw'
        else:
            flash(f'File {filename} not found', 'danger')
            return redirect(url_for('data.index'))
        
        # Return file as attachment
        return send_file(file_path, as_attachment=True)
    
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'danger')
        logger.error(f"Download error: {str(e)}", exc_info=True)
        return redirect(url_for('data.index'))

@data_bp.route('/delete/<filename>')
def delete_file(filename):
    """Delete data file"""
    try:
        # Determine if it's a raw or processed file
        if os.path.exists(os.path.join(current_app.config['PROCESSED_FOLDER'], filename)):
            file_path = os.path.join(current_app.config['PROCESSED_FOLDER'], filename)
            file_type = 'processed'
        elif os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], filename)):
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file_type = 'raw'
        else:
            flash(f'File {filename} not found', 'danger')
            return redirect(url_for('data.index'))
        
        # Delete the file
        os.remove(file_path)
        flash(f'{file_type.capitalize()} file {filename} deleted successfully', 'success')
        
    except Exception as e:
        flash(f'Error deleting file: {str(e)}', 'danger')
        logger.error(f"Delete error: {str(e)}", exc_info=True)
    
    return redirect(url_for('data.index'))

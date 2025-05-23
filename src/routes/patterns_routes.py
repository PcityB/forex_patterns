from flask import Blueprint, render_template, request, jsonify, current_app, flash, redirect, url_for, send_file
import os
import pandas as pd
import numpy as np
import json
import pickle
import logging
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
patterns_bp = Blueprint('patterns', __name__, url_prefix='/patterns')

@patterns_bp.route('/')
def index():
    """Pattern extraction dashboard"""
    # Get list of available processed data files
    processed_files = []
    if os.path.exists(current_app.config['PROCESSED_FOLDER']):
        processed_files = [f for f in os.listdir(current_app.config['PROCESSED_FOLDER']) 
                          if os.path.isfile(os.path.join(current_app.config['PROCESSED_FOLDER'], f)) 
                          and f.endswith('.csv')]
    
    # Get list of extracted pattern files
    pattern_files = []
    if os.path.exists(current_app.config['PATTERNS_FOLDER']):
        pattern_files = [f for f in os.listdir(os.path.join(current_app.config['PATTERNS_FOLDER'], 'data')) 
                        if os.path.isfile(os.path.join(current_app.config['PATTERNS_FOLDER'], 'data', f)) 
                        and f.endswith('_patterns.json')]
    
    return render_template('patterns/index.html', 
                          processed_files=processed_files,
                          pattern_files=pattern_files)

@patterns_bp.route('/extract', methods=['GET', 'POST'])
def extract_patterns():
    """Extract patterns from processed data"""
    if request.method == 'POST':
        try:
            # Get form data
            filename = request.form.get('filename')
            window_size = int(request.form.get('window_size', 10))
            step_size = int(request.form.get('step_size', 1))
            n_clusters = int(request.form.get('n_clusters', 20))
            distance_threshold = float(request.form.get('distance_threshold', 0.5))
            grid_size = int(request.form.get('grid_size', 10))
            
            # Import pattern extraction module
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from pattern_extraction import PatternExtractor
            
            # Setup paths
            input_file = os.path.join(current_app.config['PROCESSED_FOLDER'], filename)
            output_dir = os.path.join(current_app.config['PATTERNS_FOLDER'], 'data')
            viz_dir = os.path.join(current_app.config['PATTERNS_FOLDER'], 'visualizations')
            
            # Ensure directories exist
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(viz_dir, exist_ok=True)
            
            # Get timeframe from filename
            timeframe = filename.replace('_processed.csv', '').split('_')[-1]
            
            # Create extractor
            extractor = PatternExtractor(
                data_dir=os.path.dirname(input_file),
                output_dir=os.path.dirname(os.path.dirname(output_dir))
            )
            
            # Extract patterns
            flash(f'Pattern extraction started for {filename}. This may take some time...', 'info')
            
            # Run extraction with limited number of windows for web interface
            max_windows = int(request.form.get('max_windows', 5000))
            
            extractor.extract_patterns(
                timeframe=timeframe,
                window_size=window_size,
                stride=step_size,
                n_clusters=n_clusters,
                max_windows=max_windows,
                grid_rows=grid_size,
                grid_cols=grid_size
            )
            
            flash(f'Pattern extraction completed successfully for {filename}', 'success')
            return redirect(url_for('patterns.index'))
            
        except Exception as e:
            flash(f'Error extracting patterns: {str(e)}', 'danger')
            logger.error(f"Pattern extraction error: {str(e)}", exc_info=True)
            return redirect(url_for('patterns.index'))
    
    # GET request - show extraction form
    processed_files = []
    if os.path.exists(current_app.config['PROCESSED_FOLDER']):
        processed_files = [f for f in os.listdir(current_app.config['PROCESSED_FOLDER']) 
                          if os.path.isfile(os.path.join(current_app.config['PROCESSED_FOLDER'], f)) 
                          and f.endswith('.csv')]
    
    return render_template('patterns/extract.html', processed_files=processed_files)

@patterns_bp.route('/view/<filename>')
def view_patterns(filename):
    """View extracted patterns"""
    try:
        # Load pattern data
        file_path = os.path.join(current_app.config['PATTERNS_FOLDER'], 'data', filename)
        
        if not os.path.exists(file_path):
            flash(f'Pattern file {filename} not found', 'danger')
            return redirect(url_for('patterns.index'))
        
        with open(file_path, 'r') as f:
            pattern_data = json.load(f)
        
        # Get timeframe from filename
        timeframe = filename.replace('_patterns.json', '')
        
        # Check for visualization images
        viz_dir = os.path.join(current_app.config['PATTERNS_FOLDER'], 'visualizations')
        viz_files = []
        if os.path.exists(viz_dir):
            viz_files = [f for f in os.listdir(viz_dir) 
                        if os.path.isfile(os.path.join(viz_dir, f)) 
                        and f.startswith(f"{timeframe}_pattern_") 
                        and f.endswith('.png')]
        
        # Convert visualization paths to relative URLs
        viz_urls = [f"/static/patterns/visualizations/{f}" for f in viz_files]
        
        return render_template('patterns/view.html',
                              timeframe=timeframe,
                              pattern_data=pattern_data,
                              viz_urls=viz_urls)
    
    except Exception as e:
        flash(f'Error viewing patterns: {str(e)}', 'danger')
        logger.error(f"Error viewing patterns: {str(e)}", exc_info=True)
        return redirect(url_for('patterns.index'))

@patterns_bp.route('/delete/<filename>')
def delete_patterns(filename):
    """Delete pattern file"""
    try:
        # Delete pattern file
        file_path = os.path.join(current_app.config['PATTERNS_FOLDER'], 'data', filename)
        
        if not os.path.exists(file_path):
            flash(f'Pattern file {filename} not found', 'danger')
            return redirect(url_for('patterns.index'))
        
        # Get timeframe from filename
        timeframe = filename.replace('_patterns.json', '')
        
        # Delete pattern file
        os.remove(file_path)
        
        # Delete full pattern data file if exists
        full_file_path = os.path.join(current_app.config['PATTERNS_FOLDER'], 'data', f"{timeframe}_full_patterns.pkl")
        if os.path.exists(full_file_path):
            os.remove(full_file_path)
        
        # Delete visualization files
        viz_dir = os.path.join(current_app.config['PATTERNS_FOLDER'], 'visualizations')
        if os.path.exists(viz_dir):
            viz_files = [f for f in os.listdir(viz_dir) 
                        if os.path.isfile(os.path.join(viz_dir, f)) 
                        and f.startswith(f"{timeframe}_pattern_")]
            
            for viz_file in viz_files:
                os.remove(os.path.join(viz_dir, viz_file))
        
        flash(f'Pattern data for {timeframe} deleted successfully', 'success')
        
    except Exception as e:
        flash(f'Error deleting pattern data: {str(e)}', 'danger')
        logger.error(f"Error deleting pattern data: {str(e)}", exc_info=True)
    
    return redirect(url_for('patterns.index'))

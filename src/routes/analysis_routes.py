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

def restore_json_infinity(obj):
    """Convert JSON 'inf' and '-inf' strings back to float infinity"""
    if isinstance(obj, dict):
        return {k: restore_json_infinity(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [restore_json_infinity(i) for i in obj]
    elif obj == "inf":
        return float('inf')
    elif obj == "-inf":
        return -float('inf')
    return obj
    
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
analysis_bp = Blueprint('analysis', __name__, url_prefix='/analysis')

@analysis_bp.route('/')
def index():
    """Pattern analysis dashboard"""
    # Get list of available pattern files
    pattern_files = []
    if os.path.exists(os.path.join(current_app.config['PATTERNS_FOLDER'], 'data')):
        pattern_files = [f for f in os.listdir(os.path.join(current_app.config['PATTERNS_FOLDER'], 'data')) 
                        if os.path.isfile(os.path.join(current_app.config['PATTERNS_FOLDER'], 'data', f)) 
                        and f.endswith('_patterns.json')]
    
    # Get list of analysis results
    analysis_files = []
    if os.path.exists(os.path.join(current_app.config['ANALYSIS_FOLDER'], 'data')):
        analysis_files = [f for f in os.listdir(os.path.join(current_app.config['ANALYSIS_FOLDER'], 'data')) 
                         if os.path.isfile(os.path.join(current_app.config['ANALYSIS_FOLDER'], 'data', f)) 
                         and f.endswith('_analysis.json')]
    
    return render_template('analysis/index.html', 
                          pattern_files=pattern_files,
                          analysis_files=analysis_files)

@analysis_bp.route('/analyze/<filename>', methods=['GET', 'POST'])
def analyze_patterns(filename):
    """Analyze extracted patterns"""
    if request.method == 'POST':
        try:
            # Get form data
            lookahead_periods = int(request.form.get('lookahead_periods', 10))
            
            # Import pattern analysis module
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from pattern_analysis import PatternAnalyzer
            
            # Setup paths
            patterns_dir = current_app.config['PATTERNS_FOLDER']
            logger.error(patterns_dir)
            data_dir = current_app.config['PROCESSED_FOLDER']
            output_dir = current_app.config['ANALYSIS_FOLDER']
            
            # Get timeframe from filename
            timeframe = filename.replace('_patterns.json', '')
            
            # Create analyzer
            analyzer = PatternAnalyzer(patterns_dir, data_dir, output_dir)
            
            # Analyze patterns
            flash(f'Pattern analysis started for {timeframe}. This may take some time...', 'info')
            
            results = analyzer.analyze_patterns(timeframe, lookahead_periods=lookahead_periods)
            
            if results:
                flash(f'Pattern analysis completed successfully for {timeframe}', 'success')
                return redirect(url_for('analysis.view_analysis', filename=f"{timeframe}_analysis.json"))
            else:
                flash(f'Pattern analysis failed for {timeframe}', 'danger')
                return redirect(url_for('analysis.index'))
            
        except Exception as e:
            flash(f'Error analyzing patterns: {str(e)}', 'danger')
            logger.error(f"Pattern analysis error: {str(e)}", exc_info=True)
            return redirect(url_for('analysis.index'))
    
    # GET request - show analysis form
    try:
        # Get timeframe from filename
        timeframe = filename.replace('_patterns.json', '')
        
        # Check if pattern file exists
        pattern_file = os.path.join(current_app.config['PATTERNS_FOLDER'], 'data', filename)
        if not os.path.exists(pattern_file):
            flash(f'Pattern file {filename} not found', 'danger')
            return redirect(url_for('analysis.index'))
        
        # Load pattern data for display
        with open(pattern_file, 'r') as f:
            pattern_data = restore_json_infinity(json.load(f))
        
        return render_template('analysis/analyze.html', 
                              timeframe=timeframe,
                              pattern_data=pattern_data,
                              filename=filename)
    
    except Exception as e:
        flash(f'Error loading pattern data: {str(e)}', 'danger')
        logger.error(f"Error loading pattern data: {str(e)}", exc_info=True)
        return redirect(url_for('analysis.index'))

@analysis_bp.route('/view/<filename>')
def view_analysis(filename):
    """View analysis results"""
    try:
        # Load analysis data
        file_path = os.path.join(current_app.config['ANALYSIS_FOLDER'], 'data', filename)
        
        if not os.path.exists(file_path):
            flash(f'Analysis file {filename} not found', 'danger')
            return redirect(url_for('analysis.index'))
        
        with open(file_path, 'r') as f:
            analysis_data = restore_json_infinity(json.load(f))
        
        # Get timeframe from filename
        timeframe = filename.replace('_analysis.json', '')
        
        # Check for visualization images
        viz_dir = os.path.join(current_app.config['ANALYSIS_FOLDER'], 'visualizations')
        viz_files = []
        if os.path.exists(viz_dir):
            viz_files = [f for f in os.listdir(viz_dir) 
                        if os.path.isfile(os.path.join(viz_dir, f)) 
                        and (f.startswith(f"{timeframe}_") or f == "feature_importance.png" or f == "pca_clusters.png") 
                        and f.endswith('.png')]
        
        # Convert visualization paths to relative URLs
        viz_urls = {}
        for f in viz_files:
            if f == "feature_importance.png":
                viz_urls['feature_importance'] = f"/static/analysis/visualizations/{f}"
            elif f == "pca_clusters.png":
                viz_urls['pca_clusters'] = f"/static/analysis/visualizations/{f}"
            elif f.startswith(f"{timeframe}_cluster_profitability"):
                viz_urls['profitability'] = f"/static/analysis/visualizations/{f}"
        
        return render_template('analysis/view.html',
                              timeframe=timeframe,
                              analysis_data=analysis_data,
                              viz_urls=viz_urls)
    
    except Exception as e:
        flash(f'Error viewing analysis: {str(e)}', 'danger')
        logger.error(f"Error viewing analysis: {str(e)}", exc_info=True)
        return redirect(url_for('analysis.index'))

@analysis_bp.route('/delete/<filename>')
def delete_analysis(filename):
    """Delete analysis file"""
    try:
        # Delete analysis file
        file_path = os.path.join(current_app.config['ANALYSIS_FOLDER'], 'data', filename)
        
        if not os.path.exists(file_path):
            flash(f'Analysis file {filename} not found', 'danger')
            return redirect(url_for('analysis.index'))
        
        # Get timeframe from filename
        timeframe = filename.replace('_analysis.json', '')
        
        # Delete analysis file
        os.remove(file_path)
        
        # Delete visualization files
        viz_dir = os.path.join(current_app.config['ANALYSIS_FOLDER'], 'visualizations')
        if os.path.exists(viz_dir):
            viz_files = [f for f in os.listdir(viz_dir) 
                        if os.path.isfile(os.path.join(viz_dir, f)) 
                        and (f.startswith(f"{timeframe}_") or f == "feature_importance.png" or f == "pca_clusters.png")]
            
            for viz_file in viz_files:
                os.remove(os.path.join(viz_dir, viz_file))
        
        flash(f'Analysis data for {timeframe} deleted successfully', 'success')
        
    except Exception as e:
        flash(f'Error deleting analysis data: {str(e)}', 'danger')
        logger.error(f"Error deleting analysis data: {str(e)}", exc_info=True)
    
    return redirect(url_for('analysis.index'))

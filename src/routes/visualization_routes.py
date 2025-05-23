from flask import Blueprint, render_template, request, jsonify, current_app, flash, redirect, url_for, send_file
import os
import pandas as pd
import numpy as np
import json
import logging
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
visualization_bp = Blueprint('visualization', __name__, url_prefix='/visualization')

@visualization_bp.route('/')
def index():
    """Visualization dashboard"""
    # Get list of available analysis files
    analysis_files = []
    if os.path.exists(os.path.join(current_app.config['ANALYSIS_FOLDER'], 'data')):
        analysis_files = [f for f in os.listdir(os.path.join(current_app.config['ANALYSIS_FOLDER'], 'data')) 
                         if os.path.isfile(os.path.join(current_app.config['ANALYSIS_FOLDER'], 'data', f)) 
                         and f.endswith('_analysis.json')]
    
    return render_template('visualization/index.html', analysis_files=analysis_files)

@visualization_bp.route('/visualize/<filename>')
def visualize_analysis(filename):
    """Interactive visualization of analysis results"""
    try:
        # Load analysis data
        file_path = os.path.join(current_app.config['ANALYSIS_FOLDER'], 'data', filename)
        
        if not os.path.exists(file_path):
            flash(f'Analysis file {filename} not found', 'danger')
            return redirect(url_for('visualization.index'))
        
        with open(file_path, 'r') as f:
            analysis_data = json.load(f)
        
        # Get timeframe from filename
        timeframe = filename.replace('_analysis.json', '')
        
        # Generate interactive visualizations
        viz_dir = os.path.join(current_app.config['STATIC_FOLDER'], 'visualization')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create profitability chart
        if 'cluster_returns' in analysis_data:
            # Prepare data for cluster returns visualization
            clusters = []
            avg_returns = []
            win_rates = []
            
            for cluster_id, data in analysis_data['cluster_returns'].items():
                clusters.append(f"Cluster {cluster_id}")
                avg_return = data.get('avg_return', 0)
                if avg_return is None or (isinstance(avg_return, float) and (avg_return != avg_return)):  # Check for NaN
                    avg_return = 0
                avg_returns.append(avg_return * 100)  # Convert to percentage
                win_rate = data.get('win_rate', data.get('positive_rate', 0))
                if win_rate is None or (isinstance(win_rate, float) and (win_rate != win_rate)):  # Check for NaN
                    win_rate = 0
                win_rates.append(win_rate * 100)  # Convert to percentage
            
            # Create subplot with two y-axes
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add traces
            fig.add_trace(
                go.Bar(x=clusters, y=avg_returns, name="avg Return (%)", marker_color='blue'),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(x=clusters, y=win_rates, name="Win Rate (%)", marker_color='red', mode='lines+markers'),
                secondary_y=True,
            )
            
            # Set titles
            fig.update_layout(
                title_text="Cluster Profitability Analysis",
                xaxis_title="Cluster",
                barmode='group',
                height=600
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Average Return (%)", secondary_y=False)
            fig.update_yaxes(title_text="Win Rate (%)", secondary_y=True)
            
            # Save to HTML
            profitability_path = os.path.join(viz_dir, f"{timeframe}_profitability.html")
            fig.write_html(profitability_path)
        
        # Create statistical significance visualization
        if 'statistical_significance' in analysis_data:
            # Prepare data
            clusters = []
            p_values = []
            colors = []
            
            for cluster_id, data in analysis_data['statistical_significance'].items():
                clusters.append(f"Cluster {cluster_id}")
                p_values.append(data['p_value'])
                colors.append('green' if data['p_value'] < 0.05 else 'red')
            
            fig = go.Figure(data=[
                go.Bar(x=clusters, y=p_values, marker_color=colors)
            ])
            
            # Add a horizontal line at p=0.05
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=0.05,
                x1=len(clusters)-0.5,
                y1=0.05,
                line=dict(
                    color="black",
                    width=2,
                    dash="dash",
                )
            )
            
            fig.update_layout(
                title_text="Statistical Significance by Cluster (p-value)",
                xaxis_title="Cluster",
                yaxis_title="p-value",
                height=600
            )
            
            # Save to HTML
            significance_path = os.path.join(viz_dir, f"{timeframe}_significance.html")
            fig.write_html(significance_path)
        
        # Sanitize numeric values in analysis_data.profitability to avoid NaN or undefined errors in template
        if 'profitability' in analysis_data:
            profitability = analysis_data['profitability']
            for key in ['avg_return', 'win_rate', 'profit_factor']:
                value = profitability.get(key, 0)
                if value is None or (isinstance(value, float) and (value != value)):  # NaN check
                    profitability[key] = 0
            analysis_data['profitability'] = profitability

        # Sanitize cluster_returns numeric fields
        if 'cluster_returns' in analysis_data:
            for cluster_id, data in analysis_data['cluster_returns'].items():
                for key in ['avg_return', 'win_rate', 'profit_factor']:
                    value = data.get(key, 0)
                    if value is None or (isinstance(value, float) and (value != value)):
                        data[key] = 0
                analysis_data['cluster_returns'][cluster_id] = data

        return render_template('visualization/visualize.html',
                              timeframe=timeframe,
                              analysis_data=analysis_data,
                              profitability_viz=f"/static/visualization/{timeframe}_profitability.html",
                              significance_viz=f"/static/visualization/{timeframe}_significance.html")
    
    except Exception as e:
        flash(f'Error creating visualizations: {str(e)}', 'danger')
        logger.error(f"Error creating visualizations: {str(e)}", exc_info=True)
        return redirect(url_for('visualization.index'))

@visualization_bp.route('/backtest/<filename>')
def backtest_patterns(filename):
    """Backtest pattern trading strategies"""
    try:
        # Get timeframe from filename
        timeframe = filename.replace('_analysis.json', '')
        
        # Check if analysis file exists
        analysis_file = os.path.join(current_app.config['ANALYSIS_FOLDER'], 'data', filename)
        if not os.path.exists(analysis_file):
            flash(f'Analysis file {filename} not found', 'danger')
            return redirect(url_for('visualization.index'))
        
        # Load analysis data
        with open(analysis_file, 'r') as f:
            analysis_data = json.load(f)
        
        return render_template('visualization/backtest.html',
                              timeframe=timeframe,
                              analysis_data=analysis_data)
    
    except Exception as e:
        flash(f'Error loading backtest page: {str(e)}', 'danger')
        logger.error(f"Error loading backtest page: {str(e)}", exc_info=True)
        return redirect(url_for('visualization.index'))

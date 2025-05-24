import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # DON'T CHANGE THIS !!!

from flask import Flask, render_template, redirect, url_for, flash
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_for_forex_pattern_app')
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    app.config['PROCESSED_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
    app.config['PATTERNS_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'patterns')
    app.config['ANALYSIS_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'analysis')
    app.config['STATIC_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    app.config['ALLOWED_EXTENSIONS'] = {'csv', 'txt'}
    
    # Ensure directories exist
    for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'], 
                  app.config['PATTERNS_FOLDER'], app.config['ANALYSIS_FOLDER']]:
        os.makedirs(folder, exist_ok=True)
    
    # Create subdirectories for patterns and analysis
    os.makedirs(os.path.join(app.config['PATTERNS_FOLDER'], 'data'), exist_ok=True)
    os.makedirs(os.path.join(app.config['PATTERNS_FOLDER'], 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(app.config['ANALYSIS_FOLDER'], 'data'), exist_ok=True)
    os.makedirs(os.path.join(app.config['ANALYSIS_FOLDER'], 'visualizations'), exist_ok=True)
    
    # Register blueprints
    from routes.data_routes import data_bp
    from routes.patterns_routes import patterns_bp
    from routes.analysis_routes import analysis_bp
    from routes.visualization_routes import visualization_bp
    
    app.register_blueprint(data_bp)
    app.register_blueprint(patterns_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(visualization_bp)
    
    # Uncomment the following line if you need to use mysql, do not modify the SQLALCHEMY_DATABASE_URI configuration
    # app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{os.getenv('DB_USERNAME', 'root')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME', 'mydb')}"
    
    @app.route('/')
    def index():
        """Main application dashboard"""
        return render_template('index.html', current_year=datetime.now().year)
    
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def server_error(e):
        return render_template('500.html'), 500
    
    # Serve additional static folders for visualizations
    from flask import send_from_directory

    @app.route('/static/analysis/visualizations/<path:filename>')
    def analysis_visualizations(filename):
        return send_from_directory(app.config['ANALYSIS_FOLDER'] + '/visualizations', filename)

    @app.route('/static/patterns/visualizations/<path:filename>')
    def patterns_visualizations(filename):
        return send_from_directory(app.config['PATTERNS_FOLDER'] + '/visualizations', filename)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)

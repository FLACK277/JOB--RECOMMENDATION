from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
import os
from werkzeug.exceptions import BadRequest
from functools import wraps
import time

# Configuration - Use environment variables or default paths
MODEL_PATH = os.getenv('MODEL_PATH', r'C:\Users\rawat\.vscode\cli\jobrecommender\models\job_recommender_model.pkl')
JOBS_DATA_PATH = os.getenv('JOBS_DATA_PATH', r'C:\Users\rawat\.vscode\cli\jobrecommender\data\jobs_dataset.csv')

# Auto-load model on startup (set to True if you want automatic loading)
AUTO_LOAD_MODEL = os.getenv('AUTO_LOAD_MODEL', 'True').lower() == 'true'

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_json(f):
    """Decorator to validate JSON input"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        return f(*args, **kwargs)
    return decorated_function

def log_request_time(f):
    """Decorator to log request processing time"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Request to {request.endpoint} took {end_time - start_time:.3f} seconds")
        return result
    return decorated_function
class JobRecommenderSystem:
    """
    Enhanced JobRecommenderSystem that uses actual job data for recommendations
    """
    def __init__(self):
        self.model = None
        self.jobs_df = None
        self.user_interactions = None
        self.vectorizer = None
        self.similarity_matrix = None
        
    def set_jobs_data(self, jobs_df):
        """Set the jobs dataframe"""
        self.jobs_df = jobs_df
        self._prepare_recommendation_data()
        
    def _prepare_recommendation_data(self):
        """Prepare data for recommendations (vectorization, similarity calculation, etc.)"""
        if self.jobs_df is None:
            return
            
        # Example: Create text features for similarity calculation
        # Combine relevant text fields
        text_features = []
        for _, row in self.jobs_df.iterrows():
            combined_text = f"{row.get('Job Title', '')} {row.get('Key Skills', '')} {row.get('Industry', '')} {row.get('Functional Area', '')}"
            text_features.append(combined_text)
        
        # You would typically use TF-IDF or other vectorization here
        # For now, we'll use a simple approach
        self.text_features = text_features
        
    def recommend_jobs_for_user_profile(self, user_profile, top_n=5):
        """
        Generate recommendations based on user profile
        """
        if self.jobs_df is None or self.jobs_df.empty:
            return pd.DataFrame()
            
        # Extract user preferences
        user_skills = user_profile.get('Skills', '').lower()
        user_experience = int(user_profile.get('Experience', 0))
        user_industry = user_profile.get('Industry', '').lower()
        user_functional_area = user_profile.get('Functional Area', '').lower()
        user_job_title = user_profile.get('Desired Job Title', '').lower()
        
        # Calculate similarity scores for each job
        job_scores = []
        
        for idx, job in self.jobs_df.iterrows():
            score = 0
            
            # Skill matching
            job_skills = str(job.get('Key Skills', '')).lower()
            if user_skills and job_skills:
                # Simple keyword matching - you can enhance this with NLP
                user_skill_words = set(user_skills.split())
                job_skill_words = set(job_skills.split())
                skill_overlap = len(user_skill_words.intersection(job_skill_words))
                score += skill_overlap * 0.4  # 40% weight for skills
            
            # Industry matching
            job_industry = str(job.get('Industry', '')).lower()
            if user_industry and user_industry in job_industry:
                score += 0.2  # 20% weight for industry
                
            # Functional area matching
            job_func_area = str(job.get('Functional Area', '')).lower()
            if user_functional_area and user_functional_area in job_func_area:
                score += 0.2  # 20% weight for functional area
                
            # Job title matching
            job_title = str(job.get('Job Title', '')).lower()
            if user_job_title and user_job_title in job_title:
                score += 0.2  # 20% weight for job title
                
            # Experience matching (you might want to parse experience ranges)
            job_exp = str(job.get('Job Experience Required', ''))
            # Simple experience matching - enhance this based on your data format
            if 'fresher' in job_exp.lower() and user_experience <= 1:
                score += 0.1
            elif '2-5' in job_exp and 2 <= user_experience <= 5:
                score += 0.1
            elif '5+' in job_exp and user_experience >= 5:
                score += 0.1
                
            job_scores.append({
                'index': idx,
                'score': score,
                'job_data': job
            })
        
        # Sort by score and get top recommendations
        job_scores.sort(key=lambda x: x['score'], reverse=True)
        top_jobs = job_scores[:top_n]
        
        # Create recommendations dataframe
        recommendations = []
        for job_info in top_jobs:
            job = job_info['job_data']
            recommendations.append({
                'Job_ID': job.get('Job_ID', job.name),  # Use index if no Job_ID column
                'Job Title': job.get('Job Title', 'N/A'),
                'Industry': job.get('Industry', 'N/A'),
                'Functional Area': job.get('Functional Area', 'N/A'),
                'Job Experience Required': job.get('Job Experience Required', 'N/A'),
                'Key Skills': job.get('Key Skills', 'N/A'),
                'Job Salary': job.get('Job Salary', 'N/A'),
                'similarity_score': job_info['score']
            })
        
        return pd.DataFrame(recommendations)
    
    def get_collaborative_recommendations(self, user_id, top_n=5):
        """
        Generate collaborative filtering recommendations
        This is a placeholder - implement based on your user interaction data
        """
        if self.jobs_df is None or self.jobs_df.empty:
            return pd.DataFrame()
            
        # For now, return random jobs with decreasing confidence scores
        # In a real implementation, you'd use user interaction data
        import random
        
        sample_jobs = self.jobs_df.sample(min(top_n, len(self.jobs_df)))
        
        recommendations = []
        for idx, (_, job) in enumerate(sample_jobs.iterrows()):
            recommendations.append({
                'Job_ID': job.get('Job_ID', job.name),
                'Job Title': job.get('Job Title', 'N/A'),
                'Industry': job.get('Industry', 'N/A'),
                'Functional Area': job.get('Functional Area', 'N/A'),
                'confidence': 0.9 - (idx * 0.1)  # Decreasing confidence
            })
        
        return pd.DataFrame(recommendations)
    
class JobRecommendationAPI:
    def __init__(self):
        self.recommender = None
        self.jobs_df = None
        self.user_interactions_df = None
        self.is_loaded = False
        self.model_metadata = {}
    
    def load_model(self, model_path):
        """Load the job recommender system"""
        try:
            if not os.path.exists(model_path):
                # If no pickle file exists, create a new recommender instance
                logger.info("No existing model file found, creating new recommender instance")
                self.recommender = JobRecommenderSystem()
                self.is_loaded = True
                self.model_metadata = {
                    'loaded_at': datetime.now().isoformat(),
                    'model_path': 'new_instance',
                    'type': 'fresh_instance'
                }
                return True
            
            with open(model_path, 'rb') as f:
                self.recommender = pickle.load(f)
            
            self.is_loaded = True
            self.model_metadata = {
                'loaded_at': datetime.now().isoformat(),
                'model_path': model_path,
                'file_size': os.path.getsize(model_path),
                'type': 'loaded_from_pickle'
            }
            
            logger.info(f"Job recommender loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback: create new instance
            try:
                logger.info("Falling back to creating new recommender instance")
                self.recommender = JobRecommenderSystem()
                self.is_loaded = True
                self.model_metadata = {
                    'loaded_at': datetime.now().isoformat(),
                    'model_path': 'fallback_instance',
                    'type': 'fallback_instance',
                    'original_error': str(e)
                }
                return True
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
                self.is_loaded = False
                return False
    
    def load_jobs_data(self, jobs_data_path):
        """Load jobs data from CSV or pickle file"""
        try:
            if not os.path.exists(jobs_data_path):
                raise FileNotFoundError(f"Jobs data file not found: {jobs_data_path}")
            
            # Check file extension to determine how to load
            file_extension = os.path.splitext(jobs_data_path)[1].lower()
            
            if file_extension == '.csv':
                # Load CSV file with error handling
                try:
                    self.jobs_df = pd.read_csv(jobs_data_path, encoding='utf-8')
                except UnicodeDecodeError:
                    # Try with different encoding if UTF-8 fails
                    self.jobs_df = pd.read_csv(jobs_data_path, encoding='latin-1')
                logger.info(f"Jobs CSV data loaded successfully. Shape: {self.jobs_df.shape}")
                logger.info(f"CSV columns: {list(self.jobs_df.columns)}")
                
            elif file_extension == '.pkl':
                # Load pickle file
                with open(jobs_data_path, 'rb') as f:
                    self.jobs_df = pickle.load(f)
                logger.info(f"Jobs pickle data loaded successfully. Shape: {self.jobs_df.shape}")
                
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: .csv, .pkl")
            
            # Basic data validation
            if self.jobs_df.empty:
                raise ValueError("Loaded dataset is empty")
            
            # IMPORTANT: Set the jobs data in the recommender
            if self.recommender and hasattr(self.recommender, 'set_jobs_data'):
                self.recommender.set_jobs_data(self.jobs_df)
                logger.info("Jobs data set in recommender system")
            
            # Log basic info about the dataset
            logger.info(f"Dataset info: {len(self.jobs_df)} jobs loaded")
            logger.info(f"Dataset columns: {list(self.jobs_df.columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading jobs data: {str(e)}")
            return False
    
    def validate_user_profile(self, user_data):
        """Validate user profile data"""
        required_fields = ['skills', 'experience']
        missing_fields = [field for field in required_fields if field not in user_data]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate experience is numeric
        try:
            experience = int(user_data.get('experience', 0))
            if experience < 0:
                raise ValueError("Experience cannot be negative")
        except (ValueError, TypeError):
            raise ValueError("Experience must be a valid number")
        
        return True
    
    def get_recommendations(self, user_data, num_recommendations=5):
        """Generate recommendations for a user"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # Validate input
        self.validate_user_profile(user_data)
        
        if num_recommendations <= 0 or num_recommendations > 50:
            raise ValueError("num_recommendations must be between 1 and 50")
        
        try:
            # Use profile-based recommendations
            user_profile = {
                'Skills': user_data.get('skills', ''),
                'Experience': int(user_data.get('experience', 0)),
                'Role Category': user_data.get('role_category', ''),
                'Industry': user_data.get('industry', ''),
                'Functional Area': user_data.get('functional_area', ''),
                'Desired Job Title': user_data.get('job_title', '')
            }
            
            recommendations_df = self.recommender.recommend_jobs_for_user_profile(
                user_profile, 
                top_n=num_recommendations
            )
            
            if recommendations_df is None or recommendations_df.empty:
                logger.warning("No recommendations generated for user profile")
                return []
            
            # Format recommendations with better error handling
            recommendations = []
            for idx, row in recommendations_df.iterrows():
                try:
                    recommendations.append({
                        'job_id': int(row.get('Job_ID', 0)) if pd.notna(row.get('Job_ID')) else 0,
                        'job_title': str(row.get('Job Title', 'N/A')),
                        'industry': str(row.get('Industry', 'N/A')),
                        'functional_area': str(row.get('Functional Area', 'N/A')),
                        'experience_required': str(row.get('Job Experience Required', 'N/A')),
                        'key_skills': str(row.get('Key Skills', 'N/A')),
                        'salary': str(row.get('Job Salary', 'N/A')),
                        'rank': len(recommendations) + 1,
                        'match_score': float(row.get('similarity_score', 0.0)) if pd.notna(row.get('similarity_score')) else 0.0
                    })
                except Exception as e:
                    logger.warning(f"Error processing recommendation row {idx}: {str(e)}")
                    continue
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise ValueError(f"Recommendation generation failed: {str(e)}")
    
    def get_collaborative_recommendations(self, user_id, num_recommendations=5):
        """Get collaborative filtering recommendations"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        if not hasattr(self.recommender, 'get_collaborative_recommendations'):
            raise ValueError("Collaborative filtering not supported by this model")
        
        if num_recommendations <= 0 or num_recommendations > 50:
            raise ValueError("num_recommendations must be between 1 and 50")
        
        try:
            recommendations_df = self.recommender.get_collaborative_recommendations(
                user_id, top_n=num_recommendations
            )
            
            if recommendations_df is None or recommendations_df.empty:
                logger.warning(f"No collaborative recommendations found for user {user_id}")
                return []
            
            recommendations = []
            for idx, row in recommendations_df.iterrows():
                try:
                    recommendations.append({
                        'job_id': int(row.get('Job_ID', 0)) if pd.notna(row.get('Job_ID')) else 0,
                        'job_title': str(row.get('Job Title', 'N/A')),
                        'industry': str(row.get('Industry', 'N/A')),
                        'functional_area': str(row.get('Functional Area', 'N/A')),
                        'rank': len(recommendations) + 1,
                        'confidence_score': float(row.get('confidence', 0.0)) if pd.notna(row.get('confidence')) else 0.0
                    })
                except Exception as e:
                    logger.warning(f"Error processing collaborative recommendation row {idx}: {str(e)}")
                    continue
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating collaborative recommendations: {str(e)}")
            raise ValueError(f"Collaborative recommendation failed: {str(e)}")

# Initialize the recommendation system
rec_api = JobRecommendationAPI()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Job Recommendation API is running',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': rec_api.is_loaded,
        'version': '2.0.1',
        'endpoints': {
            'health': 'GET /',
            'load_model': 'POST /load-model',
            'load_default_model': 'POST /load-default-model',
            'recommend': 'POST /recommend',
            'collaborative': 'POST /recommend-collaborative',
            'model_info': 'GET /model-info',
            'data_preview': 'GET /data-preview'
        }
    })

@app.route('/data-preview', methods=['GET'])
@log_request_time
def data_preview():
    """Preview the loaded dataset structure and sample data"""
    try:
        if rec_api.jobs_df is None:
            return jsonify({'error': 'No dataset loaded'}), 400
        
        # Get basic dataset info
        dataset_info = {
            'total_rows': len(rec_api.jobs_df),
            'total_columns': len(rec_api.jobs_df.columns),
            'columns': list(rec_api.jobs_df.columns),
            'data_types': rec_api.jobs_df.dtypes.astype(str).to_dict(),
            'missing_values': rec_api.jobs_df.isnull().sum().to_dict(),
            'sample_data': rec_api.jobs_df.head(3).to_dict('records')  # First 3 rows
        }
        
        return jsonify({
            'dataset_info': dataset_info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in data_preview endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/load-default-model', methods=['POST'])
@log_request_time
def load_default_model():
    """Load model using default/configured paths"""
    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({
                'error': f'Model file not found at configured path: {MODEL_PATH}',
                'help': 'Please check the MODEL_PATH environment variable or file location'
            }), 404
        
        success = rec_api.load_model(MODEL_PATH)
        
        jobs_loaded = False
        if os.path.exists(JOBS_DATA_PATH):
            jobs_loaded = rec_api.load_jobs_data(JOBS_DATA_PATH)
        
        if success:
            return jsonify({
                'message': 'Model loaded successfully using default paths',
                'model_path': MODEL_PATH,
                'jobs_data_path': JOBS_DATA_PATH if jobs_loaded else None,
                'jobs_data_loaded': jobs_loaded,
                'timestamp': datetime.now().isoformat(),
                'model_metadata': rec_api.model_metadata
            })
        else:
            return jsonify({'error': 'Failed to load model'}), 500
            
    except Exception as e:
        logger.error(f"Error loading default model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/load-model', methods=['POST'])
@validate_json
@log_request_time
def load_model():
    """Endpoint to load the job recommendation model"""
    try:
        data = request.get_json()
        
        if not data or 'model_path' not in data:
            return jsonify({'error': 'model_path is required'}), 400
        
        model_path = data['model_path']
        jobs_data_path = data.get('jobs_data_path')
        
        success = rec_api.load_model(model_path)
        
        jobs_loaded = False
        if jobs_data_path:
            jobs_loaded = rec_api.load_jobs_data(jobs_data_path)
            if not jobs_loaded:
                logger.warning("Failed to load jobs data, but model loaded successfully")
        
        if success:
            return jsonify({
                'message': 'Model loaded successfully',
                'timestamp': datetime.now().isoformat(),
                'model_metadata': rec_api.model_metadata,
                'jobs_data_loaded': jobs_loaded
            })
        else:
            return jsonify({'error': 'Failed to load model'}), 500
            
    except Exception as e:
        logger.error(f"Error in load_model endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/recommend', methods=['POST'])
@validate_json
@log_request_time
def get_recommendations():
    """Get profile-based job recommendations"""
    try:
        if not rec_api.is_loaded:
            return jsonify({'error': 'Model not loaded. Please load model first.'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        user_data = data.get('user_profile', {})
        num_recommendations = data.get('num_recommendations', 5)
        
        # Validate num_recommendations
        try:
            num_recommendations = int(num_recommendations)
        except (ValueError, TypeError):
            return jsonify({'error': 'num_recommendations must be a valid integer'}), 400
        
        recommendations = rec_api.get_recommendations(user_data, num_recommendations)
        
        return jsonify({
            'user_id': data.get('user_id', 'anonymous'),
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'timestamp': datetime.now().isoformat(),
            'recommendation_type': 'profile_based',
            'request_params': {
                'num_recommendations': num_recommendations,
                'user_profile_fields': list(user_data.keys())
            }
        })
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in get_recommendations endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/recommend-collaborative', methods=['POST'])
@validate_json
@log_request_time
def get_collaborative_recommendations():
    """Get collaborative filtering recommendations"""
    try:
        if not rec_api.is_loaded:
            return jsonify({'error': 'Model not loaded'}), 400
        
        data = request.get_json()
        if not data or 'user_id' not in data:
            return jsonify({'error': 'user_id is required'}), 400
        
        user_id = data['user_id']
        num_recommendations = data.get('num_recommendations', 5)
        
        # Validate num_recommendations
        try:
            num_recommendations = int(num_recommendations)
        except (ValueError, TypeError):
            return jsonify({'error': 'num_recommendations must be a valid integer'}), 400
        
        recommendations = rec_api.get_collaborative_recommendations(user_id, num_recommendations)
        
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'timestamp': datetime.now().isoformat(),
            'recommendation_type': 'collaborative_filtering',
            'request_params': {
                'num_recommendations': num_recommendations
            }
        })
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in collaborative recommendations: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/model-info', methods=['GET'])
@log_request_time
def model_info():
    """Get information about the loaded model"""
    try:
        if not rec_api.is_loaded:
            return jsonify({'error': 'Model not loaded'}), 400
        
        info = {
            'model_loaded': rec_api.is_loaded,
            'model_type': 'JobRecommenderSystem',
            'model_metadata': rec_api.model_metadata,
            'has_jobs_data': rec_api.jobs_df is not None,
            'num_jobs': len(rec_api.jobs_df) if rec_api.jobs_df is not None else 0,
            'has_user_interactions': rec_api.user_interactions_df is not None,
            'supported_methods': {
                'profile_based': True,
                'collaborative_filtering': hasattr(rec_api.recommender, 'get_collaborative_recommendations') if rec_api.recommender else False
            },
            'jobs_data_columns': list(rec_api.jobs_df.columns) if rec_api.jobs_df is not None else []
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error in model_info endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Enhanced Job Recommendation API v2.0.1...")
    print("=" * 50)
    print("Available endpoints:")
    print("  GET  /                        - Health check & API info")
    print("  POST /load-model              - Load job recommendation model")
    print("  POST /load-default-model      - Load model using configured paths")
    print("  POST /recommend               - Get profile-based recommendations")
    print("  POST /recommend-collaborative - Get collaborative recommendations")
    print("  GET  /model-info              - Detailed model information")
    print("  GET  /data-preview            - Preview loaded dataset structure")
    print("=" * 50)
    
    
    # Auto-load model if enabled and files exist
    if AUTO_LOAD_MODEL:
        print(f"Auto-loading model from: {MODEL_PATH}")
        if os.path.exists(MODEL_PATH):
            success = rec_api.load_model(MODEL_PATH)
            if success:
                print("SUCCESS: Model loaded successfully")
                if os.path.exists(JOBS_DATA_PATH):
                    jobs_success = rec_api.load_jobs_data(JOBS_DATA_PATH)
                    if jobs_success:
                        print("SUCCESS: Jobs data loaded successfully")
                    else:
                        print("WARNING: Failed to load jobs data")
            else:
                print("ERROR: Failed to load model")
        else:
            print(f"WARNING: Model file not found: {MODEL_PATH}")
    else:
        print("Model auto-loading disabled. Use /load-model endpoint to load manually.")
    
    print("=" * 50)
    print("Starting server on http://0.0.0.0:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
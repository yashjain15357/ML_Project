from flask import Flask, render_template, request, jsonify
import logging
from src.logger import logging as custom_logging
from src.pipeline.predict_pipeline import CustomData, predict_pipeline
from src.exception import CustomException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Entry point
application = Flask(__name__)
app = application


@app.route('/', methods=['GET'])
def index():
    """Home page route"""
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error in index route: {str(e)}")
        return render_template('error.html', error=str(e)), 500


@app.route('/predicdata', methods=['GET', 'POST'])
def predict_datapoint():
    """Prediction form and processing route"""
    try:
        if request.method == 'GET':
            return render_template('home.html')
        
        elif request.method == 'POST':
            # Extract form data
            gender = request.form.get('gender')
            ethnicity = request.form.get('ethnicity')
            parental_education = request.form.get('parental_level_of_education')
            lunch = request.form.get('lunch')
            test_course = request.form.get('test_preparation_course')
            reading_score = request.form.get('reading_score')
            writing_score = request.form.get('writing_score')
            
            # Validate input
            if not all([gender, ethnicity, parental_education, lunch, test_course, reading_score, writing_score]):
                return render_template('home.html', error='All fields are required'), 400
            
            try:
                reading_score = float(reading_score)
                writing_score = float(writing_score)
            except ValueError:
                return render_template('home.html', error='Scores must be numeric values'), 400
            
            # Create CustomData object
            data = CustomData(
                gender=gender,
                race_ethnicity=ethnicity,
                parental_level_of_education=parental_education,
                lunch=lunch,
                test_preparation_course=test_course,
                reading_score=reading_score,
                writing_score=writing_score
            )
            
            # Get data as DataFrame
            input_data = data.get_data_as_data_frame()
            logging.info(f"Input DataFrame shape: {input_data.shape}")
            logging.info(f"Input DataFrame columns: {input_data.columns.tolist()}")
            
            # Make prediction
            predict_pipeline_obj = predict_pipeline()
            result = predict_pipeline_obj.predict(input_data)
            
            logging.info(f"Prediction result: {result[0]}")
            
            return render_template('home.html', result=result[0])
    
    except CustomException as e:
        logging.error(f"Custom exception in predict_datapoint: {str(e)}")
        return render_template('home.html', error=f"Prediction error: {str(e)}"), 500
    
    except Exception as e:
        logging.error(f"Error in predict_datapoint: {str(e)}")
        return render_template('home.html', error=f"An error occurred: {str(e)}"), 500


@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('error.html', error='Page not found'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('error.html', error='Internal server error'), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 


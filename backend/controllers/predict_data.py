from flask import Blueprint, request, jsonify
from flask_cors import CORS
from services.predict_service import PredictService
import os
import json
predict_controller = Blueprint('prediction', __name__)
predict_service = PredictService()

@predict_controller.route('/predict', methods=['GET'])
def train_model():
    try:
        
        datefrom = request.args.get('date')
        number_of_days = int(request.args.get('days'))
        model_name = request.args.get('modelName') 
        print(datefrom)
        print(number_of_days)
        
        result = predict_service.predict(datefrom, number_of_days, model_name)

        return jsonify({'success': True,'res': result})
    except Exception as e:
        print(e)
        return jsonify({'success': False, 'error': str(e)})
    
@predict_controller.route('/models', methods=['GET'])
def get_models():
    models_folder = '/home/qnchuck/Desktop/isis/backend/models'
    model_names = [file for file in os.listdir(models_folder) if file.endswith('.h5')]
    
    return jsonify(model_names)
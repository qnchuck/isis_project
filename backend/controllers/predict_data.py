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

        print(datefrom)
        print(number_of_days)
        
        result = predict_service.predict(datefrom, number_of_days)
        nested_list = result.tolist()

        # Convert the nested list to JSON
        json_data = json.dumps(nested_list)

        return jsonify({'success': True,'res': json_data})
    except Exception as e:
        print(e)
        return jsonify({'success': False, 'error': str(e)})
from flask import Blueprint, request, jsonify
from flask_cors import CORS
from services.train_service import TrainService
import os

training_controller = Blueprint('training', __name__)
train_service = TrainService()
@training_controller.route('/train_model', methods=['POST'])
def train_model():
    try:
        datefrom = request.get_json()['startDate']
        dateto = request.get_json()['endDate']
        asd = train_service.start_training(datefrom, dateto) 
        
        if asd is True:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@training_controller.route('/preprocess', methods=['POST'])
def preprocess_data():
    try:
        datefrom = request.get_json()['startDate']
        dateto = request.get_json()['endDate']
        if train_service.preprocess_data(datefrom, dateto) is True:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# my_flask_app/controllers/file_controller.py
from flask import Blueprint, request, jsonify
from flask_cors import CORS
from services.files_service import FilesService
import os


file_controller = Blueprint('file_controller', __name__)
file_service = FilesService()
@file_controller.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return {'error': 'No file part'}, 400

    file = request.files['file']

    if file_service.save_file_to_specific_directory(file):
        return {'message': 'File uploaded successfully'}, 200
    return {'error': 'Invalid file'}, 400

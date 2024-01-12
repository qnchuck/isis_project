# my_flask_app/controllers/file_controller.py
from flask import Blueprint, request, jsonify
from flask_cors import CORS
import os

file_controller = Blueprint('file_controller', __name__)

@file_controller.route('/folder_path', methods=['POST'])
def process_folder():
    try:
        print('eeseesssdadddas')
        data = request.get_json()
        folder_path = data
        print(folder_path)
        # Add your logic here to process the folder path
        # For example, you can list files in the folder:
        files = os.listdir(folder_path)

        return jsonify({'success': True, 'files': files})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


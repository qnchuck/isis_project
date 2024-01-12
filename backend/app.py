from flask import Flask, jsonify
# my_flask_app/app.py
from controllers.data_import import file_controller
from controllers.training_data import training_controller
from controllers.predict_data import predict_controller
from flask_cors import CORS

app = Flask(__name__)
app.register_blueprint(file_controller, url_prefix='/file_controller')
app.register_blueprint(training_controller, url_prefix='/training')
app.register_blueprint(predict_controller, url_prefix='/prediction')
CORS(app)

@app.route('/api/data')
def get_data():
    data = {
        'title': 'Welcome to Angular-Python App',
        'message': 'This is an example integration between Angular and Python!'
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run()


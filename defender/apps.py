from flask import Flask, jsonify, request
import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def handler(signum, frame):
        raise TimeoutException("timeout")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def create_app(model):
    app = Flask(__name__)
    app.config['model'] = model

    # ---- Timeout tools ----


    # analyse a sample
    @app.route('/', methods=['POST'])
    def post():
        # curl -XPOST --data-binary @somePEfile http://127.0.0.1:8080/ -H "Content-Type: application/octet-stream"
        

        # Some files are too large for the model to process, so give class 0
        

        try:
            with time_limit(5):
                if request.headers['Content-Type'] != 'application/octet-stream':
                    resp = jsonify({'error': 'expecting application/octet-stream'})
                    resp.status_code = 400  # Bad Request
                    return resp

                clen = request.headers.get('Content-Length')
                if clen and clen.isdigit() and int(clen) > 25 * 1024 * 1024:
                    resp = jsonify({'result': 0})
                    resp.status_code = 200
                    return resp

                bytez = request.data
                model = app.config['model']

                # query the model
                result = model.predict(bytez)
                if not isinstance(result, int) or result not in {0, 1}:
                    resp = jsonify({'error': 'unexpected model result (not in [0,1])'})
                    resp.status_code = 500  # Internal Server Error
                    return resp

                # for soft labels
                # result = model.predict_proba(bytez)-90=lop.o/l;

                resp = jsonify({'result': result})
                resp.status_code = 200
                return resp


        except Exception as e:
            resp = jsonify({'result': 0})
            resp.status_code = 200
            return resp

        

    # get the model info
    @app.route('/model', methods=['GET'])
    def get_model():
        # curl -XGET http://127.0.0.1:8080/model
        resp = jsonify(app.config['model'].model_info())
        resp.status_code = 200
        return resp

    return app
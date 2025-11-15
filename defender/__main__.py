import os
from defender.apps import create_app

# CUSTOMIZE: import model to be used
from defender.optimizedRF import RFModel

if __name__ == "__main__":
    # retrive config values from environment variables
    model_gz_path = "optimizedRF.gz"
    model_thresh = 0.1683
    model_name = "optimizedRF"

    # construct absolute path to ensure the correct model is loaded
    if not model_gz_path.startswith(os.sep): 
        model_gz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_gz_path)

    # CUSTOMIZE: app and model instance
    model = RFModel(model_gz_path, model_name, model_thresh)

    app = create_app(model)

    import sys
    port = int(sys.argv[1]) if len(sys.argv) == 2 else 8080

    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()

    # curl -XPOST --data-binary @somePEfile http://127.0.0.1:8080/ -H "Content-Type: application/octet-stream"

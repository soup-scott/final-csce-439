import json
import signal
from contextlib import contextmanager
from flask import Flask, jsonify, request
from thrember import PEFeatureExtractor

extractor = PEFeatureExtractor()
JSONL_PATH = "pe_features.jsonl"
ALLOWED_LABELS = {0, 1}

# ---- Timeout tools ----
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


# ---- Flask app ----
def create_app():
    app = Flask(__name__)

    @app.route("/", methods=["POST"])
    def post():

        if request.content_type != "application/octet-stream":
            return jsonify({"error": "expecting application/octet-stream"}), 400

        try:
            label = int(request.args.get("label", "0"))
        except ValueError:
            return jsonify({"error": "invalid label"}), 400

        if label not in ALLOWED_LABELS:
            return jsonify({
                "error": "invalid label",
                "allowed": sorted(ALLOWED_LABELS)
            }), 400

        bytez = request.data

        # ---- HARD SERVER-SIDE TIMEOUT ----
        try:
            with time_limit(5):
                raw_features = extractor.raw_features(bytez)
        except TimeoutException:
            return jsonify({"error": "extraction timed out"}), 408
        except Exception as e:
            return jsonify({
                "error": "feature extraction failed",
                "detail": f"{type(e).__name__}: {e}",
            }), 500

        # Attach label AFTER successful extraction
        raw_features["label"] = label

        # ---- Write JSONL ----
        try:
            with open(JSONL_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(raw_features) + "\n")
        except Exception as e:
            return jsonify({
                "error": "failed to write to jsonl",
                "detail": f"{type(e).__name__}: {e}",
            }), 500

        return jsonify({"status": "ok"}), 200

    return app

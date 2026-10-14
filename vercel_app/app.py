"""Flask entrypoint for Vercel and local serving of the built React interface."""

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import BadRequest, UnsupportedMediaType

try:
    from . import scoring
except ImportError:
    import scoring

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 8192


@app.after_request
def response_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/model")
def model():
    meta = scoring.load_bundle()["meta"]
    return jsonify(
        {
            key: meta[key]
            for key in (
                "model_name",
                "selection_reason",
                "test_metrics",
                "cv_metrics",
                "input_ranges",
                "reference_values",
                "dataset",
            )
        }
    )


@app.post("/api/predict")
def prediction():
    try:
        value = request.get_json()
    except (BadRequest, UnsupportedMediaType):
        return jsonify({"errors": {"request": "Send valid application/json."}}), 400
    errors = scoring.validate_inputs(value)
    if errors:
        return jsonify({"errors": errors}), 400
    return jsonify(scoring.predict(value))


@app.errorhandler(413)
def oversized(_error):
    return jsonify({"errors": {"request": "Request is too large."}}), 413


@app.get("/")
def home():
    return send_from_directory(scoring.ROOT / "public", "index.html")


@app.get("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(scoring.ROOT / "public/assets", filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

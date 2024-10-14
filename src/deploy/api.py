from flask import Flask, request, jsonify
from flask_caching import Cache

from deploy.inference import predict, load_model
from deploy.parsing import parse
from deploy.preprocessing import preprocess, scale_cryoem_blob

app = Flask(__name__)
model = load_model()
app.config["CACHE_TYPE"] = "SimpleCache"
cache = Cache(app)


@app.route("/api")
def home():
    return "Ligand Classification API is up and running!", 200


@app.route("/api/predict", methods=["POST"])
@cache.cached(timeout=300)
def classify_ligand():
    if len(request.files) == 0:
        return jsonify({"error": "No file part in the request"}), 400

    file_val = list(request.files.values())[0]

    if not (file_val.filename.endswith((".npy", ".npz", ".pts", ".xyz", ".txt", ".csv"))):
        return jsonify({"error": "Unsupported file format"}), 400

    try:
        blob = parse(file_val)

        rescale_cryoem = request.form.get("rescale_cryoem", "false").lower() == "true"
        resolution = request.form.get("resolution", None)

        if rescale_cryoem and not resolution:
            return jsonify({"error": "No resolution part in the request"}), 400

        if rescale_cryoem:
            resolution = float(resolution)
            blob = scale_cryoem_blob(blob, resolution=resolution)

        blob = preprocess(blob)

        preds = predict(blob, model)
        # preds = torch.nn.functional.softmax(preds, axis=0)
        preds = preds.to_dict("records")

        return jsonify({"predictions": preds}), 200

    except Exception as e:
        raise e
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5000, host="0.0.0.0")

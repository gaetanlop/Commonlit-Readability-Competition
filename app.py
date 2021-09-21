from flask import Flask, jsonify, request, render_template, url_for, redirect
from torch_utils import get_predictions, text_to_tensor

app = Flask(__name__)

@app.route("/", methods = ["POST", "GET"])
def home():
    if request.method == 'POST':
        text1 = request.form["nm1"]
        text2 = request.form["nm2"]

        ids1, mask1 = text_to_tensor(text1)
        prediction1 = get_predictions(ids1, mask1)
        ids2, mask2 = text_to_tensor(text2)
        prediction2 = get_predictions(ids2, mask2)
        if prediction1>prediction2:
            return render_template("index.html", prediction1 = "Text A is the easiest to read !")
        else:
            return render_template("index.html", prediction1 = "Text B is the easiest to read !")

    else:
        return render_template("index.html")        

if __name__ == '__main__':
    app.run(debug=True)
import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np

categories = ["French bulldog", "Chihuahua", "Golden retriever", "Maltese", "Miniature Dachshund", "Saint Bernard", "Shiba", "Shih tzu", "Toypoodle", "Yorkshire terrier"]
class_num = len(categories)
image_size = 64

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif"])

app = Flask(__name__)

def allowed_file(filename):
	return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model("./dog_model.h5")

graph = tf.get_default_graph()

@app.route("/", methods = ["GET", "POST"])
def upload_file():
	global graph
	with graph.as_default():
		if request.method == 'POST':
			if 'file' not in request.files:
				flash('ファイルがありません')
				return redirect(request.url)
			file = request.files['file']
			if file.filename == '':
				flash('ファイルがありません')
				return redirect(request.url)

			if file and allowed_file(file.filename):
				filename = secure_filename(file.filename)
				file.save(os.path.join(UPLOAD_FOLDER, filename))
				filepath = os.path.join(UPLOAD_FOLDER, filename)

				#受け取った画像を読み込み、np形式に変換
				img = image.load_img(filepath, target_size=(image_size, image_size))
				img = image.img_to_array(img)
				data = np.array([img])
				#変換したデータをモデルに渡して予測する
				result = model.predict(data)[0]
				predicted = result.argmax()
				pred_answer = "あなたの犬は " + categories[predicted] + " です"

				image_path = "./static/dog_img/{}.jpg".format(categories[predicted])

				#return render_template("index2.html",answer=pred_answer)
				return render_template("index2.html", answer = pred_answer, file_path =image_path)

		return render_template("index.html", answer="")

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 8080))
	app.run(host ='0.0.0.0',port = port)

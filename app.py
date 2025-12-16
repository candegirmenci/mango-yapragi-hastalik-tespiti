import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# modelin veri yolu
MODEL_PATH = 'my_model.h5'

# Sınıf isimleri
CLASS_NAMES = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew','Sooty Mould']  # <-- BURAYI GÜNCELLEYİN

print("Model yükleniyor...")
# Modeli yükle
try:
    model = load_model(MODEL_PATH)
    print("Model başarıyla yüklendi!")
except Exception as e:
    print(f"HATA: Model yüklenemedi. '{MODEL_PATH}' dosyası var mı? Hata: {e}")
    exit()

# yüklenen fotoğrafı modelde test için kullanılabilecek hale getirir
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# ana sayfa
@app.route('/', methods=['GET'])
def index():
    stats = {
        "accuracy": 98.66,
        "val_accuracy": 98.12,
        "loss": 0.0542
    }

    return render_template('index.html', stats=stats)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yok'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'})

    try:
        # Yüklenen dosyalar için klasör yoksa oluşturur
        basepath = os.path.dirname(__file__)
        uploads_dir = os.path.join(basepath, 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)

        file_path = os.path.join(uploads_dir, file.filename) # yüklediğim dosyanın nereye yazılacağını ve dosya adını tutar
        file.save(file_path) # dosyayı uploads klasörüne kaydeder

        # Tahmin kısmı
        processed_image = prepare_image(file_path) # yüklenen fotoğrafı modelin anlayacağı hale çevirir
        prediction = model.predict(processed_image) # fotoğrafın modeldeki hangi kategoriye ait olduğuna dair tahmin yapar

        predicted_class_index = np.argmax(prediction, axis=1)[0] # tahmin listesindeki en büyük sayı hangi sınıfta onun indexini verir

        # yukarıda verilen indexin gerçekten bir sınıfa karşılık gelip gelmediğini kontrol eder
        if predicted_class_index < len(CLASS_NAMES):
            predicted_class_name = CLASS_NAMES[predicted_class_index]
        else:
            predicted_class_name = f"Sınıf {predicted_class_index}"

        confidence = float(np.max(prediction)) # çıkan sonucun doğru olma ihtimalini tutar

        # tahminin hangi sınıfa ait olduğunu ve doğruluk ihtimalini ekrana yazdırır
        return jsonify({
            'class_name': predicted_class_name,
            'confidence': f"%{confidence * 100:.2f}"
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    # web sunucusunu başlat
    app.run(host='0.0.0.0', port=7860)
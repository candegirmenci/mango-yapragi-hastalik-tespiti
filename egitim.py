# Kütüphaneler
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

# ana klasörün yolu
base_dir = "/Users/flawi/PycharmProjects/görü_görüntü_web/archive_split"

# grafiklerin kaydedileceği static klasörü
static_dir = "static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# eğer veri yolu train klasörüne erişemezse hata verip kodu durdurur
if not os.path.exists(train_dir):
    print(f"HATA: {train_dir} bulunamadı. Lütfen 'base_dir' yolunu kontrol edin.")
    exit()

NUM_CLASSES = len(os.listdir(train_dir))
print(f"Sınıf Sayısı: {NUM_CLASSES}")

# Veri Çoğaltma
train_datagen = ImageDataGenerator(
    rescale=1./255, # renkleri 0 ve 1 arasına sıkıştırır
    rotation_range=30,
    width_shift_range=0.2, # yatay kaydırma
    height_shift_range=0.2, # dikey kaydırma
    shear_range=0.2, # perspektifini bozma
    zoom_range=0.2, # yakınlaştırma-uzaklaştırma
    horizontal_flip=True, # resmin simetriğini alma
    fill_mode='nearest' # kalan boşlukları en yakındaki renk değeri ile doldurur
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Model oluşturması ve eğitmesi
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x) # çok boyutlu veriyi düzleştirir
x = Dropout(0.2)(x) # eğitim sırasında rastgele verilerin %20sini kapatır ki model veriyi ezberleyemesin
predictions = Dense(NUM_CLASSES, activation='softmax')(x) # modelin yüzde kaç ihtimalle hangi sınıfa ait olduğunu söyler

model = Model(inputs=base_model.input, outputs=predictions)

# modelin optimize edilmesi
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print("Model eğitiliyor...")
history = model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator)

# eğitilmiş modeli kaydet
model.save("my_model.h5")
print("Model kaydedildi.")

# Web kısmı için eğitilen modelin grafiklerinin oluşturulması ve kaydedilmesi

# Doğruluk ve Kayıp Grafikleri
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Eğitim Accuracy')
plt.plot(epochs_range, val_acc, label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Doğruluk (Accuracy)')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Eğitim Loss')
plt.plot(epochs_range, val_loss, label='Test Loss')
plt.legend(loc='upper right')
plt.title('Kayıp (Loss)')

# Resmi static klasörüne kaydet
plt.savefig(os.path.join(static_dir, 'metrics_graph.png'))
plt.close()
print("Metrics grafiği kaydedildi.")

# 2. Confusion Matrix

steps = np.ceil(test_generator.samples / BATCH_SIZE)
y_pred_probabilities = model.predict(test_generator, steps=steps)
y_pred_labels = np.argmax(y_pred_probabilities, axis=1)
y_true_labels = test_generator.classes
class_names = list(test_generator.class_indices.keys())

cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Gerçek')
plt.xlabel('Tahmin')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(static_dir, 'confusion_matrix.png'))
plt.close()
print("Confusion Matrix kaydedildi.")

# 3. ROC Curve
y_true_bin = label_binarize(y_true_labels, classes=[*range(NUM_CLASSES)])
fpr = dict() # false positive
tpr = dict() # true positive
roc_auc = dict()

# her bir hastalık için tek tek roc eğrisi hesaplar
for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(NUM_CLASSES):
    plt.plot(fpr[i], tpr[i], label=f'Sınıf {class_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(static_dir, 'roc_curve.png'))
plt.close()
print("ROC Curve kaydedildi.")

print(f"\nİşlem Tamam! Tüm dosyalar '{static_dir}' klasörüne ve model ana dizine kaydedildi.")
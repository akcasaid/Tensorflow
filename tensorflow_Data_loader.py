import albumentations as A
import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.keras.utils import to_categorical

class DataGenerator(tf.keras.utils.Sequence):
    # Segmentasyon görevleri için geliştirilmiş. Veri üreteciyi yollar, grup boyutu, girdi boyutu vb. ile başlatır.
    def __init__(self, train_path, mask_path, transform, batch_size=32, input_size=(256,256,1), num_clases=2, shuffle=True, augmentation=False, normalization="minmax"):
        self.image_list = os.listdir(train_path)  # Eğitim yolundaki resim dosyalarının listesi
        self.batch_size = batch_size  # Eğitim için grup boyutu
        self.input_size = input_size  # Girdi resim boyutu
        self.shuffle = shuffle  # Verilerin karıştırılıp karıştırılmayacağı
        self.num_clases = num_clases  # Segmentasyon için sınıf sayısı
        self.transform = transform(train_path, mask_path, self.input_size, augmentation, num_clases, normalization)  # Uygulanacak dönüşüm
        self.step_per_epochs = len(self.image_list) // batch_size  # Her epoch için adımlar
        self.on_epoch_end()  # İlk epoch için hazırlık
    
    def on_epoch_end(self):
        # Her epoch sonunda indeksleri günceller. Gerekirse karıştırır.
        self.indexes = np.arange(len(self.image_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  # İndeksleri karıştırır
    
    def __len__(self):
        # Her epoch için grup sayısını döndürür
        return int(np.floor(len(self.image_list) / self.batch_size))
    
    def __getitem__(self, index):
        # Bir grup veri getirir
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_image_list = [self.image_list[k] for k in indexes]  # Grup için resimleri seçer
        x_train_batch, y_train_batch = self.data_generate(batch_image_list)
        return x_train_batch, y_train_batch
    
    def data_generate(self, batch_image_list):
        # Bir grup için veri üretir. Belirtilirse artırımları uygular.
        x_train_batch = np.empty((self.batch_size, *self.input_size), dtype=np.float32)
        y_train_batch = np.empty((self.batch_size, self.input_size[0], self.input_size[1], self.num_clases), dtype=np.float32)
        # Daha fazla işleme burada gerçekleşir, resimlerin yüklenmesi, dönüşümlerin uygulanması ve geri döndürülmesi dahil.


os.chdir("..")  # Mevcut çalışma dizininden bir üst dizine çıkar.
os.chdir("Teknofest Data İnceleme/")  # Mevcut çalışma dizinini "Teknofest Data İnceleme/" olarak değiştirir.
os.chdir("..")  # Yeniden bir üst dizine çıkar.


transform = A.Compose([
    A.Resize(width=416, height=416),  # Resimleri 416x416 boyutuna getirir
    A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_REPLICATE),  # Resimleri +/- 30 derece sınırları içinde döndürür
    A.HorizontalFlip(p=0.5),  # %50 olasılıkla yatay çevirme uygular
    A.VerticalFlip(p=0.1),  # %10 olasılıkla dikey çevirme uygular
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.8),  # RGB kanallarını kaydırır
    A.OneOf([
        A.Blur(blur_limit=3, p=0.2),  # Bulanıklık uygular
        A.ColorJitter(p=0.5)  # Renklerde oynama yapar
    ], p=1.0),  # Yukarıdaki dönüşümlerden birini belirli bir olasılıkla uygular
])


training = "iskemi_test_orginal"
training_mask = "iskemi_test_mask"
seg_list = []
mask_list = []
for i in os.listdir(training):
    img = cv2.imread(training + "/" + i, 0)  # Gri tonlamada resim yükler
    img = cv2.resize(img, (416, 416))  # Resmi yeniden boyutlandırır
    seg_list.append(img)  # Resmi listeye ekler
    mask = cv2.imread(training_mask + "/" + i, 0)  # İlgili maskeyi gri tonlamada yükler
    mask = cv2.resize(mask, (416, 416))  # Maskeyi yeniden boyutlandırır
    mask_list.append(mask)  # Maskeyi listeye ekler

np.save("testnew.npy", np.stack(seg_list))  # Resimleri bir NumPy dizisi olarak kaydeder
np.save("testmasknew.npy", np.stack(mask_list))  # Maskeleri bir NumPy dizisi olarak kaydeder



# YOLOv8 Rakam Tespiti ve OCR Çalışmaları

Bu repo, YOLOv8 ile rakam tespiti, RetinaNet deneyleri ve OCR (Optical Character Recognition) denemeleri için Jupyter Notebook'ları ile eğitilmiş bir model ağırlığını (`digit.pt`) içerir.

## Proje Yapısı

```
Yolov8/
├─ Digit.ipynb              # YOLOv8 ile rakam tespiti/deneyler
├─ OCR_Model.ipynb          # OCR denemeleri (metin tanıma)
├─ RetinaNet_Toplu.ipynb    # RetinaNet ile toplu deneyler
├─ digit.pt                 # Eğitilmiş YOLOv8 model ağırlığı
└─ images/
   ├─ original.jpg
   ├─ test1.jpg
   └─ test2.jpg
```

## Gereksinimler

- Python 3.9+
- Jupyter Notebook veya JupyterLab
- Paketler:
  - ultralytics (YOLOv8)
  - torch, torchvision
  - opencv-python, numpy, pandas, matplotlib, pillow
  - pytesseract (OCR için)

Hızlı kurulum (CPU):

```bash
pip install --upgrade pip
pip install ultralytics torch torchvision opencv-python numpy pandas matplotlib pillow pytesseract jupyter
```

GPU (CUDA) için PyTorch'u sisteminize uygun komutla kurun: `https://pytorch.org/get-started/locally/`

## Kullanım

### 1) Notebook'ları Çalıştırma

```bash
jupyter notebook
```
Ardından `Digit.ipynb`, `OCR_Model.ipynb` veya `RetinaNet_Toplu.ipynb` dosyalarını açıp hücreleri sırayla çalıştırın.

### 2) Eğitilmiş YOLOv8 Modeli ile Hızlı Çıkarım

Python örneği:

```python
from ultralytics import YOLO

model = YOLO('digit.pt')
results = model('images/test1.jpg')

for r in results:
    r.save()  # runs/detect/predict* klasörüne kaydeder

boxes = results[0].boxes
print(boxes.xyxy, boxes.conf, boxes.cls)
```

Komut satırı (CLI) örneği:

```bash
yolo detect predict model=digit.pt source=images/test1.jpg
yolo detect predict model=digit.pt source=images/
```

Notlar:
- Çıktılar varsayılan olarak `runs/detect/predict*` içine kaydedilir.
- Sınıf isimleri, modeli nasıl eğittiğinize göre değişebilir.

### 3) OCR Denemeleri
- `OCR_Model.ipynb` içinde OCR akış örnekleri bulunur.
- Tesseract kullanıyorsanız yerel kurulum gerekebilir ve `pytesseract.pytesseract.tesseract_cmd` ile yol tanımlanabilir.

## Eğitim (Opsiyonel)
- `Digit.ipynb` içinde eğitim/fine-tuning adımları varsa veri yollarını ve hiperparametreleri kendinize göre düzenleyin.
- Eğitim çıktıları genellikle `runs/` altında oluşturulur.

## Örnek Görseller
- `images/` klasöründeki görselleri test için kullanabilirsiniz. Kendi verilerinizi eklemek isterseniz aynı klasöre kopyalayın veya notebook/komutlarda yolu güncelleyin.

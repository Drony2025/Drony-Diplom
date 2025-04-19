# Drone Detection System

Цей проєкт містить інструменти для:
- Підготовки датасету з анотаціями.
- Навчання моделі YOLOv8 для виявлення дронів.
- Виконання детекції дронів на зображеннях та відео.
- Відстеження дронів між кадрами.

🔧 Встановлення

Перед початком роботи встанови залежності (в Google Colab або локально):

```bash
pip install kagglehub opencv-python-headless numpy pandas matplotlib pillow tqdm albumentations pyyaml ultralytics torch torchvision filterpy fpdf



Примітка: Для завантаження датасету з Kaggle потрібно API ключ (файл kaggle.json). Додай його в Google Colab або в локальне середовище.

📦 Структура
DataProcessor: завантажує, розпаковує, анотує та розбиває дані.

DroneDetector: тренує модель YOLOv8, робить інференс на зображеннях та відео.

DroneTracker: відстежує об'єкти між кадрами відео.


🚀Як запустити
1. Підготовка даних:
from drone import DataProcessor

dp = DataProcessor()
results = dp.process_dataset()


2. Навчання моделі:
from drone import DroneDetector

detector = DroneDetector()
model_path = detector.train(epochs=30, imgsz=640)


3. Виявлення дронів на зображенні:
detector.load_model(model_path)
results = detector.detect('path/to/image.jpg')
detector.visualize_detection_results(results)


4. Обробка відео:
detector.process_video('input_video.mp4', 'output_video.mp4', display=True)


5. Відстеження дронів:
from drone import DroneTracker

tracker = DroneTracker()
tracker.process_video('input_video.mp4', 'tracked_output.mp4', detector=detector)


📁 Файли

drone.py — основний модуль
data.yaml — автоматично згенерований конфіг для YOLO
results/ — результати навчання
tracking_results/ — результати трекінгу

# 🚍 Automatic Passenger Counting System

Система компьютерного зрения для автоматического учета входа/выхода пассажиров с использованием YOLO и алгоритмов трекинга.


## 📋 Особенности
- 🎯 Детекция людей с помощью YOLOv11s 
- 📍 Трекинг объектов (Bot-SORT/ByteTrack)
- 🖥️ Поддержка GPU/CPU
- 📁 Экспорт результатов в CSV


# Датасет
- Доступ к датасету из 500 видео, на котором тестируется система можно получить по [ссылке](https://drive.google.com/file/d/1zVFNOq5J2lQcRZ0p42ac5j2vli2aLEKH/view?usp=sharing). Полный датасет и методология представлены в [People Counting Dataset](https://github.com/shijieS/people-counting-dataset).
- [Датасет](https://drive.google.com/file/d/12y7UPofX-aXY-3yJRpnV3t4FfEV6FcYe/view?usp=sharing) - 7,000 аннотированных изображений для обучения модели.



# Структура проекта

```plaintext
.
├── detect/                 
│   ├── weights/             
│   │   ├── best.pt         # Лучшие веса модели
│   │   └── last.pt         # Последние веса
│   └── results.csv         # Результаты обучения
├── trackers/
│   ├── botsort.yaml        # Конфиг Bot-SORT
│   └── bytetrack.yaml      # Конфиг ByteTrack
├── test_videos/            # Тестовые видео
├── people_counter.py       # Основной скрипт
├── training.ipynb          # Обучение модели
├── testing.py              # Скрипт тестирования 
├── processed.csv           # Результат тестирования
└── metrics.ipynb           # Визуализация и метрики
```

# Основные параметры скрипта
- `model` - путь к файлу весов (.pt)
- `tracker` - путь к конфигурации трекера
- `device` - выбор CPU/GPU
- `input_path` - путь к папке для входных видео
- `y` - Y-позиция контрольной линии
- `show` - показывать ли видео в реальном времени
- `save` - сохранять ли обработанные видео
- `grey` -  преобразовывать ли видео в градации серого

# Пример кода
```python
from people_counter import PeopleCounter

counter = PeopleCounter(
    model="detect/exp/weights/best.pt",
    tracker="trackers/bytetrack.yaml")

enter, exit = counter.process_video(
    y=380,
    input_path="test_videos/1.mp4",
    output_path="result.mp4",
    show=True)
print(f"Вход: {enter}, Выход: {exit}")
```




























import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
import cv2
import random

# Используем GPU, если он доступен
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Функция для выбора активации
def get_activation_function(name):
    activations = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(0.01)
    }
    return activations.get(name, nn.ReLU())  # По умолчанию ReLU


# Определение модели - должно точно соответствовать той, которая использовалась при обучении
class FlexibleFFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, activation_name):
        super(FlexibleFFNN, self).__init__()
        self.num_layers = num_layers
        self.activation_name = activation_name

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(get_activation_function(activation_name))
        layers.append(nn.BatchNorm1d(hidden_size))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(get_activation_function(activation_name))
            layers.append(nn.BatchNorm1d(hidden_size))

        layers.append(nn.Linear(hidden_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

IMG_SIZE = 100
MARGIN = 10  # Отступ от границ
THICKNESS = 1  # Толщина обводки

def draw_circle(img):
    center = (random.randint(30, 70), random.randint(30, 70))
    radius = random.randint(10, 30)
    cv2.circle(img, center, radius, 255, THICKNESS)

def draw_square(img):
    size = random.randint(20, 40)
    x, y = random.randint(MARGIN, IMG_SIZE - size - MARGIN), random.randint(MARGIN, IMG_SIZE - size - MARGIN)
    cv2.rectangle(img, (x, y), (x + size, y + size), 255, THICKNESS)

def draw_rectangle(img):
    w, h = random.choice([(random.randint(20, 30), random.randint(40, 50)), (random.randint(40, 50), random.randint(20, 30))])  # Разные диагонали
    x, y = random.randint(MARGIN, IMG_SIZE - w - MARGIN), random.randint(MARGIN, IMG_SIZE - h - MARGIN)
    cv2.rectangle(img, (x, y), (x + w, y + h), 255, THICKNESS)

def draw_triangle(img):
    pt1 = (random.randint(10, 40), random.randint(10, 40))
    pt2 = (random.randint(20, 80), random.randint(60, 80))
    pt3 = (random.randint(60, 80), random.randint(10, 40))
    pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=255, thickness=THICKNESS)

def draw_star(img):
    center = (random.randint(30, 70), random.randint(30, 70))
    size = random.randint(15, 30)
    pts = []
    for i in range(5):
        outer = (int(center[0] + size * np.cos(2 * np.pi * i / 5)), int(center[1] + size * np.sin(2 * np.pi * i / 5)))
        inner = (int(center[0] + (size // 2) * np.cos(2 * np.pi * (i + 0.5) / 5)),
                 int(center[1] + (size // 2) * np.sin(2 * np.pi * (i + 0.5) / 5)))
        pts.extend([outer, inner])
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=255, thickness=THICKNESS)

def draw_trapezoid(img):
    x1 = random.randint(15, 35)
    x2 = x1 + random.randint(20, 30)
    y1 = random.randint(MARGIN, IMG_SIZE // 2)
    y2 = y1 + random.randint(20, 45)
    x3, x4 = x1 - random.randint(5, 15), x2 + random.randint(7, 25)
    pts = np.array([(x1, y1), (x2, y1), (x4, y2), (x3, y2)], np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=255, thickness=THICKNESS)

def draw_rhombus(img):
    center = (random.randint(40, 60), random.randint(40, 60))
    w, h = random.choice([(random.randint(10, 20), random.randint(30, 40)), (random.randint(30, 40), random.randint(10, 20))])  # Разные диагонали
    pts = np.array([(center[0], center[1] - h), (center[0] - w, center[1]), (center[0], center[1] + h),
                    (center[0] + w, center[1])], np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=255, thickness=THICKNESS)

def draw_pentagon(img):
    center = (random.randint(20, 80), random.randint(20, 80))
    size = random.randint(15, 25)
    pts = [(int(center[0] + size * np.cos(2 * np.pi * i / 5)), int(center[1] + size * np.sin(2 * np.pi * i / 5))) for i
           in range(5)]
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=255, thickness=THICKNESS)

def draw_oval(img):
    center = (random.randint(35, 65), random.randint(35, 65))
    axes = random.choice([(random.randint(15, 20), random.randint(25, 30)), ((random.randint(25, 30)), (random.randint(15, 20)))])
    cv2.ellipse(img, center, axes, 0, 0, 360, 255, THICKNESS)

def draw_semicircle(img):
    center = (random.randint(40, 60), random.randint(40, 60))
    radius = random.randint(15, 40)
    angle = random.choice([0, 90, 180, 270])  # Случайный поворот
    cv2.ellipse(img, center, (radius, radius), angle, 0, 180, 255, THICKNESS)
    if angle == 0:
        cv2.line(img, (center[0] - radius, center[1]), (center[0] + radius, center[1]), 255, THICKNESS)
    elif angle == 90:
        cv2.line(img, (center[0], center[1] - radius), (center[0], center[1] + radius), 255, THICKNESS)
    elif angle == 180:
        cv2.line(img, (center[0] - radius, center[1]), (center[0] + radius, center[1]), 255, THICKNESS)
    elif angle == 270:
        cv2.line(img, (center[0], center[1] - radius), (center[0], center[1] + radius), 255, THICKNESS)

# Список доступных фигур
shapes = {
    "Circle": draw_circle,
    "Square": draw_square,
    "Rectangle": draw_rectangle,
    "Triangle": draw_triangle,
    "Star": draw_star,
    "Trapezoid": draw_trapezoid,
    "Rhombus": draw_rhombus,
    "Pentagon": draw_pentagon,
    "Oval": draw_oval,
    "Semicircle": draw_semicircle
}


# Функция для подготовки изображения к предсказанию
def preprocess_image(img):
    # Преобразование в тензор
    img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # Нормализация к [0, 1]
    img_tensor = (img_tensor - 0.5) / 0.5  # Нормализация к [-1, 1] как в обучении
    img_tensor = img_tensor.view(1, -1)  # Преобразование в формат для нейросети: [1, 4096]
    return img_tensor


# 1. Функция для предсказания класса геометрической фигуры
def predict_shape(model, image):
    model.eval()
    with torch.no_grad():
        # Преобразуем изображение в тензор для модели
        img_tensor = preprocess_image(image).to(device)

        # Получаем предсказание
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Получаем класс с максимальной вероятностью
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = class_names[predicted_idx.item()]
        confidence = probabilities[0][predicted_idx.item()].item()

        return predicted_class, confidence, probabilities.cpu().numpy()[0]


# 2. Функция для генерации случайной фигуры и предсказания её класса
def generate_and_predict(model):
    # Генерируем случайную фигуру
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # Выбираем случайную фигуру и её функцию рисования
    shape_name = random.choice(list(shapes.keys()))
    shape_func = shapes[shape_name]

    # Рисуем фигуру
    shape_func(img)

    # Делаем предсказание
    predicted_class, confidence, class_probabilities = predict_shape(model, img)

    return img, shape_name, predicted_class, confidence, class_probabilities


# Основная функция для запуска программы
def main():
    # Параметры модели (должны соответствовать параметрам обученной модели)
    input_size = IMG_SIZE * IMG_SIZE  # 64x64 = 4096 пикселей
    hidden_size = 2048
    num_classes = 10  # Количество классов (фигур)
    num_layers = 5  # Количество скрытых слоев
    activation = 'leaky_relu'  # Функция активации

    # Список классов (должен соответствовать классам, использованным при обучении)
    global class_names
    class_names = ['Circle', 'Oval', 'Pentagon', 'Rectangle', 'Rhombus',
                   'Semicircle', 'Square', 'Star', 'Trapezoid', 'Triangle']

    # Создаем экземпляр модели
    model = FlexibleFFNN(input_size, hidden_size, num_classes, num_layers, activation)
    model.to(device)

    # Загружаем веса модели
    model_path = "results1\model_layers5_leaky_relu.pth"  # Путь к сохраненным весам
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Модель успешно загружена из {model_path}")
    except:
        print(f"Не удалось загрузить модель из {model_path}. Убедитесь, что файл существует.")
        return

    # Интерактивный режим
    while True:
        print("\nВыберите действие:")
        print("1. Сгенерировать случайную фигуру и предсказать её класс")
        print("2. Сгенерировать несколько случайных фигур и оценить точность")
        print("3. Выйти")

        choice = input("Введите номер действия (1-3): ")

        if choice == '1':
            # Генерируем и предсказываем одну фигуру
            img, true_shape, predicted_shape, confidence, probabilities = generate_and_predict(model)

            # Выводим результаты
            plt.figure(figsize=(10, 6))

            # Изображение
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.title(
                f"Истинный класс: {true_shape}\nПредсказанный класс: {predicted_shape}\nУверенность: {confidence:.2f}")
            plt.axis('off')

            # Вероятности классов
            plt.subplot(1, 2, 2)
            plt.bar(class_names, probabilities)
            plt.title('Вероятности классов')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

        elif choice == '2':
            # Генерируем и предсказываем несколько фигур
            num_samples = int(input("Введите количество изображений для генерации: "))

            correct = 0
            predictions = []

            # Генерируем и предсказываем
            for i in range(num_samples):
                img, true_shape, predicted_shape, confidence, _ = generate_and_predict(model)
                predictions.append((img, true_shape, predicted_shape, confidence))

                if true_shape == predicted_shape:
                    correct += 1

            accuracy = correct / num_samples * 100
            print(f"\nТочность на {num_samples} случайных изображениях: {accuracy:.2f}%")

            # Отображаем некоторые примеры
            n_examples = min(num_samples, 9)
            plt.figure(figsize=(12, 12))

            for i in range(n_examples):
                img, true_shape, predicted_shape, confidence = predictions[i]
                plt.subplot(3, 3, i + 1)
                plt.imshow(img, cmap='gray')
                color = 'green' if true_shape == predicted_shape else 'red'
                plt.title(f"Истинный: {true_shape}\nПредсказан: {predicted_shape}\nУверен.: {confidence:.2f}",
                          color=color)
                plt.axis('off')

            plt.tight_layout()
            plt.show()

        elif choice == '3':
            print("Программа завершена.")
            break

        else:
            print("Неверный выбор. Пожалуйста, введите число от 1 до 3.")


if __name__ == "__main__":
    main()
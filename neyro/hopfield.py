import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score


class HopfieldNetwork:
    def __init__(self, input_size):
        """
        Инициализация сети Хопфилда
        :param input_size: размер входного образа (количество пикселей)
        """
        self.input_size = input_size
        # Инициализация нулевой весовой матрицы
        self.weights = np.zeros((input_size, input_size))
        # Сохранение оригинальных образов
        self.memory_patterns = []
        self.labels = []

    def train(self, patterns, labels=None):
        """
        Обучение сети Хопфилда по правилу Хебба
        :param patterns: список образов для запоминания
        :param labels: метки образов (необязательно)
        """
        self.memory_patterns = patterns.copy()
        if labels is not None:
            self.labels = labels.copy()

        # Количество образов
        n_patterns = len(patterns)
        print(f"Обучение сети на {n_patterns} образах...")

        # Инициализация весовой матрицы
        self.weights = np.zeros((self.input_size, self.input_size))

        # Правило Хебба для создания весовой матрицы
        for pattern in tqdm(patterns):
            # Преобразуем в вектор {-1, 1}
            pattern_vector = pattern.flatten() * 2 - 1
            # Внешнее произведение вектора на самого себя
            outer_product = np.outer(pattern_vector, pattern_vector)
            # Обнуляем диагональные элементы (нет самосвязи)
            np.fill_diagonal(outer_product, 0)
            # Добавляем к весовой матрице
            self.weights += outer_product

        # Нормализация весов
        self.weights /= n_patterns

        print("Обучение завершено.")

    def recall(self, pattern, max_iterations=500, threshold=0):
        """
        Восстановление полного образа из искаженного
        :param pattern: искаженный образ
        :param max_iterations: максимальное количество итераций
        :param threshold: порог активации
        :return: восстановленный образ
        """
        # Преобразуем в вектор {-1, 1}
        pattern_vector = pattern.flatten() * 2 - 1

        # Итеративный процесс восстановления
        for i in range(max_iterations):
            # Сохраняем предыдущее состояние для проверки сходимости
            prev_pattern = pattern_vector.copy()

            # Асинхронное обновление (в случайном порядке)
            indices = list(range(self.input_size))
            random.shuffle(indices)

            for idx in indices:
                # Вычисление входного сигнала для нейрона
                activation = np.dot(self.weights[idx], pattern_vector)
                # Пороговая функция активации
                pattern_vector[idx] = 1 if activation > threshold else -1

            # Проверка сходимости
            if np.array_equal(pattern_vector, prev_pattern):
                break

        # Преобразуем обратно в {0, 1} и восстанавливаем исходную форму
        return ((pattern_vector + 1) / 2).reshape(pattern.shape)

    def recognize(self, pattern):
        """
        Распознавание образа: определение, к какому из запомненных образов он ближе всего
        :param pattern: входной образ
        :return: индекс ближайшего образа и расстояние до него
        """
        if len(self.memory_patterns) == 0:  # Fixed this line
            return None, float('inf')

        # Получаем восстановленный образ
        recalled_pattern = self.recall(pattern)

        # Находим ближайший из запомненных образов
        min_distance = float('inf')
        best_match_idx = -1

        for i, mem_pattern in enumerate(self.memory_patterns):
            distance = np.sum(np.abs(recalled_pattern.flatten() - mem_pattern.flatten()))
            if distance < min_distance:
                min_distance = distance
                best_match_idx = i

        return best_match_idx, min_distance

    def predict(self, pattern):
        """
        Предсказание метки для входного образа
        :param pattern: входной образ
        :return: предсказанная метка
        """
        if not self.labels:
            return None

        # Распознаем образ
        idx, _ = self.recognize(pattern)
        if idx is not None:
            return self.labels[idx]
        return None

    def energy(self, pattern):
        """
        Расчет энергии Хопфилда для образа
        :param pattern: образ для расчета энергии
        :return: значение энергии
        """
        pattern_vector = pattern.flatten() * 2 - 1
        energy = -0.5 * np.dot(np.dot(pattern_vector, self.weights), pattern_vector)
        return energy


# Функции для предобработки изображений
def load_and_preprocess_images(directory, img_size=(100, 100), max_images=50):
    """
    Загрузка и предобработка изображений из директории
    :param directory: путь к директории с изображениями
    :param img_size: размер для масштабирования изображений
    :param max_images: максимальное количество изображений каждого класса
    :return: список бинаризованных изображений и список меток
    """
    images = []
    labels = []

    # Перебираем подпапки (классы)
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"Загрузка класса: {class_name}")
        count = 0

        # Перебираем файлы в подпапке
        for filename in os.listdir(class_dir):
            if count >= max_images:
                break

            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, filename)

                # Загрузка изображения
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Масштабирование
                img = cv2.resize(img, img_size)

                # Бинаризация (порог Отсу)
                _, binary_img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                images.append(binary_img)
                labels.append(class_name)
                count += 1

    return np.array(images), labels


def add_noise(image, noise_level=0.1):
    """
    Добавление шума к изображению
    :param image: исходное изображение
    :param noise_level: доля пикселей, которые будут изменены
    :return: зашумленное изображение
    """
    noisy_image = np.copy(image)
    num_pixels = noisy_image.size
    num_noise_pixels = int(noise_level * num_pixels)

    # Выбираем случайные индексы для шума
    flat_indices = np.random.choice(num_pixels, num_noise_pixels, replace=False)

    # Инвертируем значения в выбранных пикселях
    flat_image = noisy_image.flatten()
    flat_image[flat_indices] = 1 - flat_image[flat_indices]

    return flat_image.reshape(noisy_image.shape)


def create_partial_image(image, coverage=0.7):
    """
    Создание частичного изображения (часть пикселей скрыта)
    :param image: исходное изображение
    :param coverage: доля видимых пикселей
    :return: частичное изображение
    """
    partial_image = np.copy(image)
    num_pixels = partial_image.size
    num_hidden_pixels = int((1 - coverage) * num_pixels)

    # Выбираем случайные индексы для скрытия
    flat_indices = np.random.choice(num_pixels, num_hidden_pixels, replace=False)

    # Устанавливаем случайные значения для скрытых пикселей
    flat_image = partial_image.flatten()
    flat_image[flat_indices] = np.random.randint(0, 2, size=len(flat_indices))

    return flat_image.reshape(partial_image.shape)


# Функции визуализации
def plot_patterns(patterns, titles=None, figsize=(15, 5)):
    """
    Визуализация набора образов
    :param patterns: список образов
    :param titles: список заголовков
    :param figsize: размер фигуры
    """
    n = len(patterns)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.imshow(patterns[i], cmap='binary')
        ax.axis('off')
        if titles and i < len(titles):
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()


def evaluate_network(network, test_patterns, test_labels, noise_levels=[0.1, 0.2, 0.3], partial_coverages=[0.8, 0.6]):
    """
    Оценка работы сети на тестовых данных с разными уровнями шума и покрытия
    :param network: обученная сеть Хопфилда
    :param test_patterns: тестовые образы
    :param test_labels: метки тестовых образов
    :param noise_levels: уровни шума для тестирования
    :param partial_coverages: доля видимых пикселей для тестирования
    """
    results = {}

    # Тестирование на оригинальных образах
    predictions = []
    for pattern in test_patterns:
        pred = network.predict(pattern)
        predictions.append(pred)

    acc = accuracy_score(test_labels, predictions)
    results['original'] = acc
    print(f"Точность на оригинальных образах: {acc:.4f}")

    # Тестирование на зашумленных образах
    for noise_level in noise_levels:
        noisy_predictions = []
        for pattern in test_patterns:
            noisy_pattern = add_noise(pattern, noise_level)
            pred = network.predict(noisy_pattern)
            noisy_predictions.append(pred)

        acc = accuracy_score(test_labels, noisy_predictions)
        results[f'noise_{noise_level}'] = acc
        print(f"Точность при уровне шума {noise_level}: {acc:.4f}")

    # Тестирование на частичных образах
    for coverage in partial_coverages:
        partial_predictions = []
        for pattern in test_patterns:
            partial_pattern = create_partial_image(pattern, coverage)
            pred = network.predict(partial_pattern)
            partial_predictions.append(pred)

        acc = accuracy_score(test_labels, partial_predictions)
        results[f'partial_{coverage}'] = acc
        print(f"Точность при покрытии {coverage}: {acc:.4f}")

    return results


def plot_energy_landscape(network, pattern, iterations=10, noise_levels=np.linspace(0, 1, 10)):
    """
    Визуализация изменения энергии при восстановлении образа
    :param network: обученная сеть Хопфилда
    :param pattern: исходный образ
    :param iterations: количество итераций для каждого уровня шума
    :param noise_levels: уровни шума
    """
    energies = []

    for noise_level in noise_levels:
        noisy_pattern = add_noise(pattern, noise_level)
        pattern_energies = []

        # Сохраняем начальную энергию
        current_pattern = noisy_pattern.copy()
        energy = network.energy(current_pattern)
        pattern_energies.append(energy)

        # Выполняем несколько итераций восстановления и считаем энергию
        for _ in range(iterations):
            current_pattern = network.recall(current_pattern, max_iterations=1)
            energy = network.energy(current_pattern)
            pattern_energies.append(energy)

        energies.append(pattern_energies)

    # Визуализация
    plt.figure(figsize=(10, 6))
    for i, noise_level in enumerate(noise_levels):
        plt.plot(range(iterations + 1), energies[i], label=f'Шум {noise_level:.1f}')

    plt.xlabel('Итерация')
    plt.ylabel('Энергия')
    plt.title('Изменение энергии в процессе восстановления образа')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_experiment(img_size=(100, 100), max_train_images=50, max_test_images=50):
    """
    Запуск полного эксперимента
    :param img_size: размер изображения (уменьшенный для сети Хопфилда)
    :param max_train_images: максимальное количество образов каждого класса для обучения
    :param max_test_images: максимальное количество образов каждого класса для тестирования
    """
    print(f"Запуск эксперимента с изображениями размером {img_size[0]}x{img_size[1]}")

    # Загрузка обучающих данных
    train_patterns, train_labels = load_and_preprocess_images('dataset/train', img_size, max_train_images)
    print(f"Загружено {len(train_patterns)} обучающих образов")

    # Загрузка тестовых данных
    test_patterns, test_labels = load_and_preprocess_images('dataset/test', img_size, max_test_images)
    print(f"Загружено {len(test_patterns)} тестовых образов")

    # Инициализация и обучение сети Хопфилда
    network = HopfieldNetwork(img_size[0] * img_size[1])
    network.train(train_patterns, train_labels)

    # Визуализация некоторых обучающих образов
    print("Примеры обучающих образов:")
    indices = np.random.choice(len(train_patterns), min(5, len(train_patterns)), replace=False)
    selected_patterns = [train_patterns[i] for i in indices]
    selected_labels = [train_labels[i] for i in indices]
    plot_patterns(selected_patterns, selected_labels)

    # Демонстрация восстановления из зашумленных образов
    print("Демонстрация восстановления из зашумленных образов:")
    for i in range(min(3, len(train_patterns))):
        pattern = train_patterns[i]
        noisy_pattern = add_noise(pattern, 0.2)
        recalled_pattern = network.recall(noisy_pattern)

        plot_patterns([pattern, noisy_pattern, recalled_pattern],
                      [f"Оригинал ({train_labels[i]})", "С шумом (20%)", "Восстановленный"])

    # Демонстрация восстановления из частичных образов
    print("Демонстрация восстановления из частичных образов:")
    for i in range(min(3, len(train_patterns))):
        pattern = train_patterns[i]
        partial_pattern = create_partial_image(pattern, 0.7)
        recalled_pattern = network.recall(partial_pattern)

        plot_patterns([pattern, partial_pattern, recalled_pattern],
                      [f"Оригинал ({train_labels[i]})", "Частичный (70%)", "Восстановленный"])

    # Оценка работы сети
    print("Оценка работы сети:")
    results = evaluate_network(network, test_patterns, test_labels)

    # Визуализация изменения энергии
    print("Визуализация изменения энергии:")
    plot_energy_landscape(network, train_patterns[0])

    return network, results


if __name__ == "__main__":
    # Запуск эксперимента с уменьшенным размером изображений
    # (классическая сеть Хопфилда ограничена в своей емкости)
    network, results = run_experiment(img_size=(100, 100))
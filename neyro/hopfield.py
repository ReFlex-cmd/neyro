import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm  # Для индикатора прогресса

# --- Параметры ---
# Используем размер из твоего датасета
IMG_SIZE = 100
# Рассчитываем количество нейронов
N_NEURONS = IMG_SIZE * IMG_SIZE
# Укажи ТОЧНЫЙ путь к твоему датасету
DATASET_PATH = r"C:\Users\ReFlex\Documents\piton\neyro\dataset"  # Используем raw string r"..." для путей Windows

# Список фигур соответствует папкам в твоем датасете
SHAPE_NAMES = [
    "Circle", "Square", "Rectangle", "Triangle", "Star",
    "Trapezoid", "Rhombus", "Pentagon", "Oval", "Semicircle"
]


# --- Функции для работы с данными ---

def binarize_image(img_path, img_size=IMG_SIZE, threshold=127):
    """
    Загружает изображение, изменяет размер (если нужно),
    конвертирует в оттенки серого, выравнивает в вектор
    и бинаризует в биполярный формат (-1 для фона, 1 для фигуры).
    """
    try:
        # Загружаем в оттенках серого
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # print(f"Предупреждение: Не удалось загрузить изображение: {img_path}. Пропускаем.")
            return None  # Возвращаем None, если файл не найден или поврежден

        # Изменение размера (на всякий случай, если в датасете есть другие размеры)
        if img.shape[0] != img_size or img.shape[1] != img_size:
            img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        else:
            img_resized = img

        # Бинаризация и преобразование в биполярный вектор (-1, 1)
        # Пиксели > threshold становятся 1, остальные -1
        bipolar_vector = np.where(img_resized.flatten() > threshold, 1, -1)
        return bipolar_vector

    except Exception as e:
        print(f"Ошибка при обработке файла {img_path}: {e}")
        return None


def vector_to_image(vector, img_size=IMG_SIZE):
    """Преобразует биполярный вектор (-1, 1) обратно в изображение (0, 255)."""
    # Проверка значений в векторе (для отладки)
    unique_values = np.unique(vector)
    print(f"Уникальные значения в векторе: {unique_values}")

    # Значения > 0 становятся 255 (белый), остальные 0 (черный)
    img_flat = np.where(vector > 0, 255, 0).astype(np.uint8)
    return img_flat.reshape((img_size, img_size))


def add_noise(vector, noise_level):
    """
    Добавляет шум к биполярному вектору (-1, 1),
    инвертируя случайную часть его элементов.
    noise_level: доля элементов для инверсии (0.0 до 1.0).
    """
    if noise_level == 0:
        return vector.copy()

    noisy_vector = vector.copy()
    num_elements = len(vector)
    num_flips = int(noise_level * num_elements)

    # Выбираем случайные уникальные индексы для инверсии
    flip_indices = random.sample(range(num_elements), num_flips)

    # Инвертируем значения по выбранным индексам (-1 -> 1, 1 -> -1)
    noisy_vector[flip_indices] *= -1
    return noisy_vector


# --- Класс Сети Хопфилда ---

class HopfieldNetwork:
    def __init__(self, num_neurons):
        """Инициализация сети."""
        self.num_neurons = num_neurons
        # Матрица весов, инициализирована нулями
        self.weights = np.zeros((num_neurons, num_neurons), dtype=np.float32)
        # Список для хранения эталонных паттернов (для отладки или анализа)
        self.memories = []
        print(f"Сеть Хопфилда создана с {num_neurons} нейронами.")

    def train(self, patterns):
        """
        Обучает сеть (вычисляет веса) на основе списка эталонных биполярных паттернов.
        Используется правило Хебба.
        patterns: список numpy-массивов, где каждый массив - биполярный вектор эталона.
        """
        num_patterns = len(patterns)
        if num_patterns == 0:
            print("Ошибка: Нет паттернов для обучения.")
            return

        print(f"Обучение сети на {num_patterns} эталонных паттернах...")
        # Сохраняем копии эталонов
        self.memories = np.array(patterns)

        # Применяем правило Хебба: W = sum(p * p^T) for p in patterns
        # (считаем, что p - вектор-столбец)
        # В NumPy проще через внешнее произведение векторов-строк
        for p in tqdm(patterns, desc="Вычисление весов"):
            # Убедимся, что p - это 1D массив правильной формы
            p_vector = p.reshape(1, -1)  # Форма (1, N)
            # Добавляем внешнее произведение к матрице весов
            self.weights += np.outer(p_vector, p_vector)

        # Деление на число паттернов необязательно, но может помочь с масштабом
        # self.weights /= num_patterns
        self.weights /= len(patterns)  # Нормализация по количеству паттернов

        # Важно: обнуляем диагональные веса (нейрон не должен влиять сам на себя)
        np.fill_diagonal(self.weights, 0)

        # Добавляем небольшой коэффициент регуляризации
        lambda_reg = 0.001
        self.weights *= (1 - lambda_reg)

        print("Обучение завершено. Матрица весов рассчитана.")

    def recall(self, pattern, max_iter=200, threshold=0.0, stop_on_convergence=True, verbose=False):
        """
        Восстанавливает эталон из входного (возможно, зашумленного) паттерна.
        Выполняет итеративное обновление состояния сети.
        pattern: входной биполярный вектор.
        max_iter: максимальное количество итераций обновления.
        threshold: СКОРРЕКТИРОВАННЫЙ порог активации нейронов для спарсенных образов.
        stop_on_convergence: останавливаться ли, когда состояние перестает меняться.
        verbose: выводить ли промежуточную информацию.
        """
        current_state = pattern.copy().astype(np.float32)

        if verbose:
            print(
                f"Начало восстановления. Итераций: {max_iter}, Скорректированный порог: {threshold}")  # Изменено сообщение

        for i in range(max_iter):
            prev_state = current_state.copy()

            # # --- Синхронное обновление ---
            # activation = np.dot(current_state, self.weights)
            #
            # # *** Ключевое изменение: Используем ненулевой порог ***
            # current_state = np.where(activation > threshold, 1, -1).astype(np.float32)

            indices = np.random.permutation(self.num_neurons)
            for idx in indices:
                # Вычисляем активацию для одного нейрона
                activation = np.dot(current_state, self.weights[:, idx])
                # Обновляем состояние одного нейрона
                current_state[idx] = 1 if activation > threshold else -1

            # --- Проверка на сходимость ---
            if stop_on_convergence and np.array_equal(current_state, prev_state):
                if verbose:
                    print(f"Сеть сошлась на итерации {i + 1}.")
                return current_state

            if verbose and (i + 1) % 10 == 0:
                print(f"  Итерация {i + 1}/{max_iter} завершена.")

        if verbose:
            if stop_on_convergence:
                print(f"Сеть не сошлась за {max_iter} итераций. Возвращается последнее состояние.")
            else:
                print(f"Достигнут лимит итераций ({max_iter}). Возвращается последнее состояние.")

        return current_state

    def calculate_energy(self, state):
        """Вычисляет энергию Ляпунова для данного состояния сети."""
        # E = -0.5 * sum_i(sum_j(W_ij * s_i * s_j))
        # В матричном виде: E = -0.5 * s^T * W * s
        # Убедимся, что state это вектор-строка для правильного умножения
        s = state.reshape(1, -1)
        energy = -0.5 * np.dot(s, np.dot(self.weights, s.T))
        return energy.item()  # Возвращаем скалярное значение


# --- Основной блок исполнения ---

if __name__ == "__main__":

    # 1. Проверка пути к датасету
    if not os.path.isdir(DATASET_PATH):
        print(f"Ошибка: Директория датасета не найдена по пути: {DATASET_PATH}")
        print("Пожалуйста, проверьте параметр DATASET_PATH в коде.")
        exit()

    # --- Загрузка эталонных образов для обучения ---
    # Будем использовать по одному "чистому" образу каждой фигуры из папки train
    train_patterns = []
    print(f"\n--- Загрузка эталонов из {os.path.join(DATASET_PATH, 'train')} ---")
    loaded_shapes = []  # Чтобы отслеживать, какие фигуры загружены

    for shape_name in SHAPE_NAMES:
        shape_train_dir = os.path.join(DATASET_PATH, "train", shape_name)
        if not os.path.isdir(shape_train_dir):
            print(f"Предупреждение: Папка для эталона '{shape_name}' не найдена: {shape_train_dir}")
            continue

        # Пытаемся найти первое изображение .png в папке
        found_image = False
        for filename in sorted(os.listdir(shape_train_dir)):  # Сортируем для воспроизводимости
            if filename.lower().endswith('.png'):
                img_path = os.path.join(shape_train_dir, filename)
                pattern_vector = binarize_image(img_path)
                if pattern_vector is not None:
                    train_patterns.append(pattern_vector)
                    loaded_shapes.append(shape_name)
                    print(f"  Загружен эталон для: {shape_name} (из файла {filename})")
                    found_image = True
                    break  # Берем только первый найденный файл
        if not found_image:
            print(f"Предупреждение: Не найдено .png файлов для эталона '{shape_name}' в {shape_train_dir}")

    if not train_patterns:
        print("\nОшибка: Не удалось загрузить ни одного эталонного образа для обучения.")
        exit()

    print(f"\nЗагружено {len(train_patterns)} эталонных паттернов для обучения.")

    # --- Создание и обучение сети Хопфилда ---
    hopfield_net = HopfieldNetwork(num_neurons=N_NEURONS)
    hopfield_net.train(train_patterns)

    # --- Тестирование и Визуализация ---
    print("\n--- Тестирование восстановления из зашумленных образов ---")
    num_test_examples = len(loaded_shapes)  # Тестируем по одному примеру для каждой загруженной фигуры
    noise_level_test = 0.1  # Уровень шума для тестовых изображений (30%)

    plt.figure(figsize=(num_test_examples * 3, 9))  # Фигура для всех примеров
    plot_idx = 1  # Индекс для subplot

    for i, shape_name in enumerate(loaded_shapes):
        print(f"\nТестирование фигуры: {shape_name}")

        # Загрузка тестового образа (можно взять другой файл из папки test)
        shape_test_dir = os.path.join(DATASET_PATH, "test", shape_name)
        test_image_path = None
        if os.path.isdir(shape_test_dir):
            # Ищем первый .png файл в тестовой папке
            for filename in sorted(os.listdir(shape_test_dir)):
                if filename.lower().endswith('.png'):
                    test_image_path = os.path.join(shape_test_dir, filename)
                    print(f"  Используется тестовый файл: {filename}")
                    break

        if test_image_path is None:
            print(
                f"  Предупреждение: Не найден тестовый .png файл для {shape_name}. Используем тренировочный эталон для теста.")
            # Если нет тестового, используем сам эталон для демонстрации зашумления/восстановления
            original_pattern = hopfield_net.memories[i]  # Берем из запомненных
            test_image_path = f"Тренировочный эталон {shape_name}"  # Для заголовка
        else:
            original_pattern = binarize_image(test_image_path)
            if original_pattern is None:
                print(
                    f"  Ошибка: Не удалось загрузить тестовый файл {test_image_path}. Пропускаем фигуру {shape_name}.")
                continue  # Переходим к следующей фигуре

        # Добавляем шум к тестовому образу
        noisy_pattern = add_noise(original_pattern, noise_level_test)

        # Восстановление с помощью сети Хопфилда
        print(f"  Запуск восстановления из зашумленного образа (шум {noise_level_test * 100:.0f}%)...")
        recalled_pattern = hopfield_net.recall(noisy_pattern, max_iter=100, threshold=0, verbose=False)
        # Преобразование векторов обратно в изображения для визуализации
        original_image = vector_to_image(original_pattern)
        noisy_image = vector_to_image(noisy_pattern)
        recalled_image = vector_to_image(recalled_pattern)

        # После восстановления и перед визуализацией
        # np.save(f"recalled_pattern_{shape_name}.npy", recalled_pattern)

        # Визуализация: Оригинал | Зашумленный | Восстановленный
        # Оригинал (Тестовый)
        plt.subplot(3, num_test_examples, plot_idx)
        plt.imshow(original_image, cmap='gray')
        plt.title(f"{shape_name}\nОригинал")
        plt.axis('off')

        # Зашумленный
        plt.subplot(3, num_test_examples, plot_idx + num_test_examples)
        plt.imshow(noisy_image, cmap='gray')
        plt.title(f"Шум: {noise_level_test * 100:.0f}%")
        plt.axis('off')

        # Восстановленный
        plt.subplot(3, num_test_examples, plot_idx + 2 * num_test_examples)
        plt.imshow(recalled_image, cmap='gray')
        plt.title("Восстановлено")
        plt.axis('off')

        # Альтернативный способ визуализации
        # plt.subplot(3, num_test_examples, plot_idx + 2 * num_test_examples)
        # recalled_reshaped = recalled_pattern.reshape((IMG_SIZE, IMG_SIZE))
        # plt.imshow(recalled_reshaped, cmap='bwr', vmin=-1, vmax=1)
        # plt.title("Восстановлено (raw)")
        # plt.axis('off')

        plot_idx += 1  # Переходим к следующей колонке

        # Сравнение восстановленного с эталоном (опционально)
        stored_memory = hopfield_net.memories[i]
        diff = np.sum(recalled_pattern != stored_memory)
        similarity = 100 * (1 - diff / N_NEURONS)
        print(f"  Восстановленный образ совпадает с эталоном на {similarity:.2f}% ({diff} отличий из {N_NEURONS})")
        print(
            f"  Энергия: Оригинал={hopfield_net.calculate_energy(original_pattern):.1f}, Шум={hopfield_net.calculate_energy(noisy_pattern):.1f}, Восст={hopfield_net.calculate_energy(recalled_pattern):.1f}")

    # Финальная настройка и отображение графиков
    plt.suptitle(f'Восстановление {len(loaded_shapes)} фигур сетью Хопфилда ({IMG_SIZE}x{IMG_SIZE})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Оставляем место для suptitle
    plt.show()

    print("\n--- Работа программы завершена ---")

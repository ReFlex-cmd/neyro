import time
import torch
import os
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2 # <--- Импортируем OpenCV

# ... (parse_args остается без изменений) ...

# Класс для загрузки датасета геометрических фигур с бинаризацией Оцу и НОРМАЛИЗАЦИЕЙ
class GeometricShapesDataset(Dataset):
    def __init__(self, root_dir, img_size, transform=None, split='train'):
        self.root_dir = os.path.join(root_dir, split)
        self.img_size = img_size # Сохраняем размер изображения
        # Трансформации: Resize до нормализации, ToTensor после
        self.resize_transform = transforms.Resize((self.img_size, self.img_size))
        self.tensor_transform = transforms.ToTensor()
        self.transform = transform # Дополнительные трансформации ПОСЛЕ ToTensor (если нужны)
        self.classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]) # Убедимся, что это директории
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        print(f"Сканирование {self.root_dir}...")
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Директория датасета не найдена: {self.root_dir}")

        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            img_count = 0
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.samples.append((img_path, self.class_to_idx[cls]))
                    img_count += 1
                # else:
                #    print(f"Предупреждение: Пропуск не-изображения или не-файла: {img_path}")
            print(f"  Класс '{cls}': найдено {img_count} изображений.")
        print(f"Всего найдено {len(self.samples)} валидных изображений.")
        if len(self.samples) == 0:
             print("ПРЕДУПРЕЖДЕНИЕ: В датасете не найдено ни одного изображения!")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('L')  # Преобразуем в оттенки серого
        except Exception as e:
            print(f"Ошибка загрузки изображения {img_path}: {e}")
            # Вернуть тензор нулей правильной формы
            return torch.zeros((1, self.img_size, self.img_size)), -1 # Метка ошибки

        # --- Предобработка ---
        # 1. Изменить размер
        image = self.resize_transform(image)

        # 2. Преобразовать в NumPy массив (uint8)
        img_np = np.array(image)

        # 3. Бинаризация методом Оцу
        # Возвращает бинарное изображение (0 или 255)
        _, thresh_img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # --- Нормализация положения и ориентации ---
        img_normalized = None
        try:
            # 4. Вычисление моментов бинарного изображения
            moments = cv2.moments(thresh_img_np) # Не нужно binaryImage=True, т.к. вход уже бинарный

            # 5. Обработка случая пустого изображения (все пиксели фона)
            if moments["m00"] < 1e-5: # Используем небольшой порог для стабильности
                # Если объект отсутствует, оставляем как есть (черный квадрат)
                # или можно вернуть исходное бинарное изображение
                img_normalized = thresh_img_np
                # print(f"Предупреждение: m00 близок к нулю для {img_path}. Пропуск нормализации.")
            else:
                # 6. Вычисление центра масс
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

                # 7. Центрирование: Вычисление матрицы сдвига и применение
                img_h, img_w = thresh_img_np.shape
                center_x, center_y = img_w // 2, img_h // 2
                tx = center_x - cx
                ty = center_y - cy
                translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                # Применяем сдвиг к ИСХОДНОМУ бинарному изображению
                img_centered = cv2.warpAffine(thresh_img_np, translation_matrix, (img_w, img_h),
                                              flags=cv2.INTER_NEAREST, # Важно для бинарных!
                                              borderMode=cv2.BORDER_CONSTANT, # Заполняем фон
                                              borderValue=0) # Черным (или 255 если фон белый)

                # 8. Вычисление угла ориентации (используем центральные моменты mu)
                # Центральные моменты не меняются при сдвиге, можно использовать из исходных 'moments'
                mu20 = moments['mu20']
                mu02 = moments['mu02']
                mu11 = moments['mu11']

                # Угол главной оси (в радианах), затем в градусах
                # Добавим малое значение к знаменателю для избежания деления на ноль, если mu20 == mu02
                angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02 + 1e-5)
                angle_deg = np.degrees(angle_rad)

                # 9. Поворот для выравнивания: Вычисление матрицы поворота и применение
                # Поворачиваем вокруг нового центра изображения (center_x, center_y)
                rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1.0)
                # Применяем поворот к УЖЕ ЦЕНТРИРОВАННОМУ изображению
                img_normalized = cv2.warpAffine(img_centered, rotation_matrix, (img_w, img_h),
                                                flags=cv2.INTER_NEAREST, # Важно для бинарных!
                                                borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=0)

        except Exception as e_norm:
            print(f"Ошибка нормализации для {img_path}: {e_norm}. Используется ненормализованное бинарное изображение.")
            # В случае любой ошибки нормализации, используем просто бинаризованное
            img_normalized = thresh_img_np

        # --- Финальное преобразование ---
        # 10. Преобразовать нормализованное бинарное изображение (0 или 255) в тензор (0.0 или 1.0)
        # tensor_transform (ToTensor) ожидает HxW или HxWxC, наш img_normalized это HxW (uint8)
        # ToTensor автоматически делит на 255.0 и меняет порядок на CxHxW
        image_tensor = self.tensor_transform(img_normalized)

        # 11. Применяем доп. трансформации, если они есть (маловероятно после этого)
        if self.transform:
             image_tensor = self.transform(image_tensor)

        return image_tensor, label

# ... (retrieve, retrieve_async (если используется), noise_* функции остаются) ...

def main():
    args = parse_args()
    t_global = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Устройство: {device}, классов: {args.num_classes}, прототипов на класс: {args.K}')

    t0 = time.time()
    # Основной transform теперь None, т.к. все внутри Dataset
    transform = None

    # Передаем img_size в Dataset
    try:
        train_dataset = GeometricShapesDataset(root_dir='dataset', img_size=args.img_size, transform=transform, split='train')
        test_dataset = GeometricShapesDataset(root_dir='dataset', img_size=args.img_size, transform=transform, split='test')
    except FileNotFoundError as e:
        print(e)
        return # Прерываем выполнение, если датасет не найден

    # Проверка на пустые датасеты после инициализации
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Ошибка: Тренировочный или тестовый датасет пуст после инициализации.")
        print("Пожалуйста, проверьте пути и содержимое папок 'dataset/train' и 'dataset/test'.")
        return

    # Используем DataLoader
    # Установите num_workers > 0 для ускорения, но 0 проще для отладки OpenCV
    # pin_memory=True может ускорить передачу на GPU, если используется CUDA
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False,
                             num_workers=0, pin_memory=torch.cuda.is_available())

    print("Загрузка данных из DataLoader...")
    try:
        train_imgs, train_lbls = next(iter(train_loader))
        test_imgs, test_lbls = next(iter(test_loader))
    except StopIteration:
        print("Ошибка: Не удалось загрузить данные из DataLoader.")
        return
    except RuntimeError as e_load:
        # Часто возникает из-за проблем с разделяемой памятью при num_workers > 0
        print(f"Ошибка DataLoader: {e_load}")
        print("Попробуйте запустить с num_workers=0.")
        return

    print("Фильтрация ошибок загрузки...")
    # Отфильтруем возможные ошибки загрузки/обработки (где label == -1)
    valid_train_idx = train_lbls != -1
    train_imgs = train_imgs[valid_train_idx]
    train_lbls = train_lbls[valid_train_idx]

    valid_test_idx = test_lbls != -1
    test_imgs = test_imgs[valid_test_idx]
    test_lbls = test_lbls[valid_test_idx]

    if train_imgs.numel() == 0 or test_imgs.numel() == 0:
        print("Ошибка: Не осталось валидных изображений после фильтрации.")
        return

    print(f"Используется {len(train_lbls)} тренировочных и {len(test_lbls)} тестовых валидных изображений.")

    train_lbls = train_lbls.to(device)
    test_lbls = test_lbls.to(device)
    # Изображения N x 1 x H x W после DataLoader, убираем канал цвета (squeeze)
    # и преобразуем в вектор N x (H*W)
    train_imgs = train_imgs.squeeze(1).view(len(train_lbls), -1).to(device)
    test_imgs = test_imgs.squeeze(1).view(len(test_lbls), -1).to(device)

    print(f'[Данные подготовлены с нормализацией] {time.time() - t0:.2f}s')
    print(f'Размерность вектора изображения (N): {train_imgs.shape[1]}') # Добавил вывод размерности

    # 2. Построение прототипов:
    t0 = time.time()
    prototypes = []
    proto_lbls = []
    unique_labels, counts = torch.unique(train_lbls, return_counts=True)
    print("Распределение классов в трейне:")
    for lbl, count in zip(unique_labels, counts):
         if lbl.item() < len(train_dataset.classes): # Проверка индекса
              print(f"  Класс {train_dataset.classes[lbl.item()]} ({lbl.item()}): {count.item()} шт.")
         else:
              print(f"  Неизвестный класс {lbl.item()}: {count.item()} шт.")


    for c in range(args.num_classes):
        # Ищем индексы для текущего класса c
        idxs = (train_lbls == c).nonzero(as_tuple=False).view(-1)

        if len(idxs) == 0:
            if c < len(train_dataset.classes):
                 print(f"Предупреждение: Класс '{train_dataset.classes[c]}' ({c}) не найден в валидном тренировочном наборе. Пропуск прототипов.")
            else:
                 print(f"Предупреждение: Класс с индексом {c} (ожидался) не найден.")
            continue # Пропускаем этот класс, если нет примеров

        # Перемешиваем найденные индексы
        perm = torch.randperm(len(idxs), device=device)
        # Выбираем минимум из доступных образцов и запрошенного K
        k_actual = min(args.K, len(idxs))
        sel = idxs[perm[:k_actual]]

        print(f"  Класс {train_dataset.classes[c]}: Выбираем {k_actual} из {len(idxs)} прототипов.")

        for i in sel:
            p = train_imgs[i] # Вектор 0.0 / 1.0
            # Биполяризация: > 0.5 -> +1, иначе (т.е. 0.0) -> -1
            b = torch.where(p > 0.5, torch.tensor(1., device=device), torch.tensor(-1., device=device))
            prototypes.append(b)
            proto_lbls.append(c)

    if not prototypes:
         print("Ошибка: Не удалось создать ни одного прототипа! Проверьте данные и параметр K.")
         return

    P = torch.stack(prototypes)  # (patterns, N)
    proto_lbls = torch.tensor(proto_lbls, device=device)
    print(f'[Прототипы] {P.shape[0]} шт. собрано за {time.time() - t0:.2f}s')

    # 3. Построение матрицы весов: псевдообратное правило
    t0 = time.time()
    # Проверка на случай, если P слишком мала для pinverse
    if P.shape[0] == 0:
        print("Ошибка: Нет прототипов для построения матрицы весов.")
        return
    elif P.shape[0] > P.shape[1]:
        print(f"Предупреждение: Количество прототипов ({P.shape[0]}) больше размерности вектора ({P.shape[1]}). "
              f"Емкость сети может быть превышена.")

    try:
        # Вычисляем W = P^T * pinv(P * P^T) * P
        M = P @ P.T  # (patterns x patterns)
        # Добавим небольшое значение к диагонали для стабильности pinverse, если нужно
        # M.add_(torch.eye(M.shape[0], device=device) * 1e-5)
        M_inv = torch.pinverse(M)
        W = P.T @ M_inv @ P  # (N x N)
        W.fill_diagonal_(0) # Обнуляем диагональ
        print(f'[Матрица W] псевдообратное правило за {time.time() - t0:.2f}s')
    except torch.linalg.LinAlgError as e_pinv:
        print(f"Ошибка вычисления псевдообратной матрицы: {e_pinv}")
        print("Возможно, матрица M сингулярна или плохо обусловлена.")
        print("Попробуйте уменьшить K или проверить прототипы на линейную зависимость.")
        return


    # 4. Подготовка тестовой выборки
    t0 = time.time()
    # Используем все доступные тестовые изображения после фильтрации
    n_test = len(test_lbls)
    if args.test_samples > 0 and args.test_samples < n_test:
        print(f"Ограничиваем тест {args.test_samples} случайными образцами из {n_test}")
        perm = torch.randperm(n_test, device=device)
        sel = perm[:args.test_samples]
        testP_vectors = test_imgs[sel]
        testL = test_lbls[sel]
        # Сохраним оригинальные НЕвекторизованные изображения для визуализации
        # Нужно будет загрузить их снова или изменить DataLoader для тестов
    else:
        print(f"Используем все {n_test} тестовых образцов.")
        testP_vectors = test_imgs # Это уже векторы 0.0/1.0
        testL = test_lbls

    print(f'[Тест] {len(testL)} образцов подготовлено за {time.time() - t0:.2f}s')

    # --- Список шумовых функций и уровней ---
    noise_funcs = {
        'BitFlip': noise_bit_flip,
        'Gaussian': noise_gaussian,
        'Dropout': noise_dropout
    }
    noise_levels = np.linspace(0, 0.5, 6) # 0%, 10%, 20%, 30%, 40%, 50%
    results = {name: [] for name in noise_funcs}

    # --- 5. Оценка точности при разных видах шума ---
    # Используем синхронное обновление по умолчанию
    retrieve_func = retrieve # Или retrieve_async, если хотите его использовать
    #retrieve_func = lambda state, W, max_iter: retrieve_async(state, W, max_iter=50) # Пример с async

    print("Начало оценки точности...")
    for name, func in noise_funcs.items():
        t_start = time.time()
        level_accuracies = []
        for level in noise_levels:
            preds = []
            correct_count = 0
            total_count = 0
            # Итерация по тестовым ВЕКТОРАМ
            for i in range(len(testP_vectors)):
                x_vector = testP_vectors[i] # Вектор 0.0 / 1.0
                true_label = testL[i].item()

                # Биполяризация тестового входа: >0.5 -> +1, 0.0 -> -1
                inp_bipolar = torch.where(x_vector > 0.5, torch.tensor(1., device=device), torch.tensor(-1., device=device))

                # Добавление шума к биполярному вектору
                noisy_bipolar = func(inp_bipolar, level)

                # Восстановление из зашумленного состояния
                out_bipolar = retrieve_func(noisy_bipolar, W, max_iter=args.max_iter)

                # Классификация: находим ближайший прототип к восстановленному вектору
                # P: (num_prototypes, N), out_bipolar: (N) -> sims: (num_prototypes)
                sims = P @ out_bipolar
                if torch.isnan(sims).any():
                     print(f"Предупреждение: NaN в схожести для теста {i}, шум {name}@{level}. Пропуск предсказания.")
                     pred_label = -1 # Ошибка предсказания
                else:
                    best_proto_idx = torch.argmax(sims).item()
                    pred_label = proto_lbls[best_proto_idx].item() # Получаем метку класса предсказанного прототипа

                preds.append(pred_label)
                if pred_label == true_label:
                    correct_count += 1
                total_count += 1

            # Расчет точности для данного уровня шума
            if total_count > 0:
                acc = correct_count / total_count
            else:
                acc = 0.0
            level_accuracies.append(acc)
            print(f"  {name} @ {level*100:.0f}%: Acc={acc*100:.2f}%")

        results[name] = level_accuracies # Сохраняем список точностей для всех уровней
        print(f'[Eval {name}] {time.time() - t_start:.2f}s')


    # --- 6. Текстовый вывод точностей ---
    print('\nИтоговые точности при разных видах шума:')
    print('Уровень шума: Точность')
    header = "Шум     | " + " | ".join([f"{l*100:4.0f}%" for l in noise_levels])
    print(header)
    print("-" * len(header))
    for name, accs in results.items():
        acc_str = " | ".join([f"{acc*100:4.1f}" for acc in accs])
        print(f"{name:<8}| {acc_str}")

    # --- 7. Построение графика точности по шуму ---
    plt.figure(figsize=(10, 6))
    for name, accs in results.items():
        plt.plot(noise_levels * 100, accs, marker='o', label=name) # Умножаем на 100 для оси X в %
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('Точность')
    plt.title(f'Робастность сети Хопфилда (Нормализация + Otsu, K={args.K})')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05) # Ось Y от 0 до 105%
    plt.xticks(noise_levels * 100) # Явные метки на оси X
    plt.tight_layout()
    plt.savefig('accuracy_noise_chart_normalized.png')
    print("\nГрафик точности сохранен в 'accuracy_noise_chart_normalized.png'")
    # plt.show() # Раскомментируйте, если хотите сразу увидеть график

    # --- 8. Визуализация примеров восстановления ---
    n_display = min(args.display_samples, len(testL))
    if n_display > 0:
        print(f"\nВизуализация {n_display} примеров восстановления...")
        # Для визуализации нам нужны оригинальные ИЗОБРАЖЕНИЯ, а не векторы
        # Перезагрузим несколько тестовых образцов (или изменим логику выше)
        vis_indices = torch.randperm(len(test_dataset))[:n_display].tolist() # Случайные индексы из ВСЕГО тест датасета

        noise_level_vis = 0.2 # Уровень шума для визуализации (20%)

        for noise_name, noise_func in noise_funcs.items():
            plt.figure(figsize=(12, 4 * n_display // 3 + 2 )) # Адаптируем размер
            plt.suptitle(f'Восстановление: Нормализация+Otsu, Шум: {noise_name} @ {noise_level_vis*100:.0f}%', fontsize=14)

            for i, original_idx in enumerate(vis_indices):
                # Загружаем ОДНО изображение с нормализацией и бинаризацией
                vis_img_tensor, vis_label_idx = test_dataset[original_idx] # Получаем тензор 0.0/1.0 (1, H, W)
                vis_label_name = test_dataset.classes[vis_label_idx] if vis_label_idx != -1 else "Ошибка"

                if vis_label_idx == -1: continue # Пропускаем ошибки загрузки/обработки

                # Оригинальное Нормализованное Бинарное изображение для показа
                orig_norm_binary_img = vis_img_tensor.squeeze(0).cpu().numpy() # (H, W) со значениями 0.0 или 1.0

                # Создаем биполярный ВЕКТОР для сети
                vis_vector_float = vis_img_tensor.view(-1) # (H*W)
                inp_bipolar = torch.where(vis_vector_float > 0.5, torch.tensor(1.), torch.tensor(-1.)).to(device)

                # Добавляем шум
                noisy_bipolar = noise_func(inp_bipolar, noise_level_vis)
                noisy_img = noisy_bipolar.cpu().view(args.img_size, args.img_size).numpy() # Для визуализации

                # Восстановление
                rec_bipolar = retrieve_func(noisy_bipolar, W, max_iter=args.max_iter)
                rec_img = rec_bipolar.cpu().view(args.img_size, args.img_size).numpy() # Для визуализации

                # Классификация восстановленного
                sims = P @ rec_bipolar
                if torch.isnan(sims).any():
                    pred_label_name = "NaN"
                else:
                    best_proto_idx = torch.argmax(sims).item()
                    pred_label_idx = proto_lbls[best_proto_idx].item()
                    pred_label_name = test_dataset.classes[pred_label_idx]

                # Отображение
                row = i // 3
                col = i % 3
                ax_idx = i * 3 + 1 # Индекс для subplot (начинается с 1)

                ax = plt.subplot(n_display, 3, ax_idx)
                ax.imshow(orig_norm_binary_img, cmap='gray') # Показываем нормализованное 0/1
                ax.set_title(f'Норм. ({vis_label_name})')
                ax.axis('off')

                ax = plt.subplot(n_display, 3, ax_idx + 1)
                ax.imshow(noisy_img, cmap='gray', vmin=-1, vmax=1) # Биполярное
                ax.set_title(f'Шум {noise_level_vis*100:.0f}%')
                ax.axis('off')

                ax = plt.subplot(n_display, 3, ax_idx + 2)
                ax.imshow(rec_img, cmap='gray', vmin=-1, vmax=1) # Биполярное
                color = 'green' if pred_label_name == vis_label_name else 'red'
                ax.set_title(f'Восст. ({pred_label_name})', color=color)
                ax.axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            plt.savefig(f'restoration_normalized_{noise_name}.png')
            print(f"Визуализация для шума '{noise_name}' сохранена.")
            # plt.show() # Раскомментируйте, если хотите сразу увидеть

    print(f'\n[Всего времени] {time.time() - t_global:.2f}s')


if __name__ == '__main__':
    # Установим seed для воспроизводимости случайных операций (выбор прототипов, шум, ...)
    # torch.manual_seed(42)
    # np.random.seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(42)
    main()
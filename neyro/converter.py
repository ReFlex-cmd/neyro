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
import cv2

# Разбор аргументов командной строки
def parse_args():
    parser = argparse.ArgumentParser(description='Улучшенная сеть Хопфилда для геометрических фигур')
    parser.add_argument('--num_classes', type=int, default=10, help='Количество классов фигур')
    parser.add_argument('--K', type=int, default=10, help='Число прототипов на класс')
    parser.add_argument('--test_samples', type=int, default=100, help='Число тестовых образцов для оценки')
    parser.add_argument('--display_samples', type=int, default=9, help='Число изображений для визуализации')
    parser.add_argument('--max_iter', type=int, default=30, help='Максимум итераций в извлечении')
    parser.add_argument('--img_size', type=int, default=100, help='Размер изображения после ресайза')
    return parser.parse_args()


# Класс для загрузки датасета геометрических фигур с бинаризацией Оцу
class GeometricShapesDataset(Dataset):
    def __init__(self, root_dir, img_size, transform=None, split='train'): # Добавим img_size для Resize перед Otsu
        self.root_dir = os.path.join(root_dir, split)
        # Разделим transform: нам нужно изменить размер *до* Оцу, а ToTensor *после*
        self.resize_transform = transforms.Resize((img_size, img_size))
        self.tensor_transform = transforms.ToTensor() # Стандартный ToTensor
        self.transform = transform # Сохраняем для возможных доп. трансформаций ПОСЛЕ ToTensor
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            # Проверяем, что это директория
            if not os.path.isdir(cls_dir):
                print(f"Предупреждение: {cls_dir} не является директорией, пропуск.")
                continue
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                # Проверяем, что это файл изображения (простая проверка по расширению)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((img_path, self.class_to_idx[cls]))
                else:
                     print(f"Предупреждение: Пропуск не-изображения или не-файла: {img_path}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('L')  # Преобразуем в оттенки серого
        except Exception as e:
            print(f"Ошибка загрузки изображения {img_path}: {e}")
            # Вернуть пустое изображение или другое значение по умолчанию
            # Важно: нужно чтобы размерность совпадала!
            # Создадим черный квадрат нужного размера
            image = Image.new('L', (self.resize_transform.size[0], self.resize_transform.size[1]), 0)
            label = -1 # Или другая метка для ошибки

        # 1. Изменить размер с помощью PIL/Torchvision transform
        image = self.resize_transform(image)

        # 2. Преобразовать в NumPy массив для OpenCV
        img_np = np.array(image)

        # 3. Применить метод Оцу
        # cv2.threshold возвращает порог (ret) и бинаризованное изображение (thresh_img)
        # Мы используем cv2.THRESH_BINARY (пиксели > порога -> maxval, иначе 0)
        # и cv2.THRESH_OTSU, чтобы порог определялся автоматически
        # Используем 255 как максимальное значение для uint8 изображений
        _, thresh_img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Преобразовать обратно в PIL Image (необязательно, но удобно для ToTensor)
        # Или можно сразу создать тензор из NumPy
        # image_otsu_pil = Image.fromarray(thresh_img_np)

        # 5. Преобразовать бинаризованное изображение (0 или 255) в тензор (0.0 или 1.0)
        # Используем unsqueeze(0), т.к. ToTensor ожидает HxWxC или HxW, а у нас HxW
        image_tensor = self.tensor_transform(thresh_img_np) # ToTensor сам делит на 255.0

        # Применяем доп. трансформации, если они есть (маловероятно после Otsu)
        if self.transform:
             image_tensor = self.transform(image_tensor)

        return image_tensor, label


# Функция динамики сети Хопфилда (извлечение)
def retrieve(state, W, max_iter=10):
    s = state.clone()
    for _ in range(max_iter):
        new = torch.sign(W @ s)
        new[new == 0] = 1
        if torch.equal(new, s):
            break
        s = new
    return s


# Шумовые функции
# 1. Переворот случайных битов (бит-флип)
def noise_bit_flip(p, level):
    p2 = p.clone()
    num = p2.numel()
    n_flip = int(level * num)
    idx = torch.randperm(num, device=p.device)[:n_flip]
    flat = p2.view(-1)
    flat[idx] *= -1
    return flat.view(p2.shape)


# 2. Гауссовский шум + порог
def noise_gaussian(p, level):
    # level — относительная дисперсия
    p2 = p.clone().view(-1)
    noise = torch.randn_like(p2) * level
    v = p2.float() + noise
    # порог по нулю: >0 -> +1, <=0 -> -1
    return torch.where(v >= 0, torch.tensor(1., device=p.device), torch.tensor(-1., device=p.device)).view(p.shape)


# 3. Пропуск пикселей (dropout/частичное скрытие)
def noise_dropout(p, level):
    p2 = p.clone().view(-1)
    num = p2.numel()
    n_drop = int(level * num)
    idx = torch.randperm(num, device=p.device)[:n_drop]
    flat = p2.clone()
    flat[idx] = 1  # заполняем +1 (фон)
    return flat.view(p.shape)


# Основная функция
def main():
    args = parse_args()
    t_global = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Устройство: {device}, классов: {args.num_classes}, прототипов на класс: {args.K}')

    t0 = time.time()
    # Убираем Resize и ToTensor из основного transform, т.к. они теперь внутри Dataset
    # Можно оставить другие трансформации, если нужны *после* ToTensor
    transform = None # Или transforms.Compose([... другие трансформы ...])

    # Передаем img_size в Dataset
    train_dataset = GeometricShapesDataset(root_dir='dataset', img_size=args.img_size, transform=transform, split='train')
    test_dataset = GeometricShapesDataset(root_dir='dataset', img_size=args.img_size, transform=transform, split='test')

    # Проверка на пустые датасеты
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Ошибка: Тренировочный или тестовый датасет пуст. Проверьте пути и содержимое папок.")
        return

    # Используем DataLoader с num_workers=0 для отладки, можно увеличить для скорости
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)

    try:
        train_imgs, train_lbls = next(iter(train_loader))
        test_imgs, test_lbls = next(iter(test_loader))
    except StopIteration:
        print("Ошибка: Не удалось загрузить данные из DataLoader. Возможно, датасеты пусты.")
        return

    # Отфильтруем возможные ошибки загрузки (где label == -1)
    valid_train_idx = train_lbls != -1
    train_imgs = train_imgs[valid_train_idx]
    train_lbls = train_lbls[valid_train_idx]

    valid_test_idx = test_lbls != -1
    test_imgs = test_imgs[valid_test_idx]
    test_lbls = test_lbls[valid_test_idx]

    if train_imgs.numel() == 0 or test_imgs.numel() == 0:
        print("Ошибка: Не осталось валидных изображений после фильтрации ошибок загрузки.")
        return


    train_lbls = train_lbls.to(device)
    test_lbls = test_lbls.to(device)
    # Изображения уже N x 1 x H x W после DataLoader, убираем канал цвета
    train_imgs = train_imgs.squeeze(1).view(-1, args.img_size * args.img_size).to(device)
    test_imgs = test_imgs.squeeze(1).view(-1, args.img_size * args.img_size).to(device)

    print(f'[Загрузка данных c Otsu] {time.time() - t0:.2f}s')
    print(f'Загружено {len(train_dataset)} тренировочных и {len(test_dataset)} тестовых изображений')
    print(f'Классы: {train_dataset.classes}')

    # 2. Построение прототипов: K случайных примеров на класс
    t0 = time.time()
    prototypes = []
    proto_lbls = []
    for c in range(args.num_classes):
        idxs = (train_lbls == c).nonzero(as_tuple=False).view(-1)
        if len(idxs) == 0:
            print(f"Предупреждение: Класс {c} не найден в тренировочном наборе")
            continue
        perm = torch.randperm(len(idxs), device=device)
        # Выбираем минимум из доступных образцов и запрошенного K
        k_actual = min(args.K, len(idxs))
        sel = idxs[perm[:k_actual]]
        for i in sel:
            p = train_imgs[i]
            b = torch.where(p > 0.5, torch.tensor(1., device=device), torch.tensor(-1., device=device))
            prototypes.append(b)
            proto_lbls.append(c)

    P = torch.stack(prototypes)  # (patterns, N)
    proto_lbls = torch.tensor(proto_lbls, device=device)
    print(f'[Прототипы] {P.shape[0]} шт. собрано за {time.time() - t0:.2f}s')

    # 3. Построение матрицы весов: псевдообратное правило
    t0 = time.time()
    M = P @ P.T  # (patterns x patterns)
    M_inv = torch.pinverse(M)
    W = P.T @ M_inv @ P  # (N x N)
    W.fill_diagonal_(0)
    print(f'[Матрица W] псевдообратное правило за {time.time() - t0:.2f}s')

    # 4. Подготовка тестовой выборки
    t0 = time.time()
    mask = torch.ones_like(test_lbls, dtype=torch.bool)  # Берем все классы
    idxs = mask.nonzero(as_tuple=False).view(-1)
    perm = torch.randperm(len(idxs), device=device)
    # Выбираем минимум из доступных образцов и запрошенного кол-ва
    n_test = min(args.test_samples, len(idxs))
    sel = idxs[perm[:n_test]]
    testP = test_imgs[sel]
    testL = test_lbls[sel]
    print(f'[Тест] {len(testL)} образцов отобрано за {time.time() - t0:.2f}s')

    # Список шумовых функций и уровней
    noise_funcs = {
        'BitFlip': noise_bit_flip,
        'Gaussian': noise_gaussian,
        'Dropout': noise_dropout
    }
    noise_levels = np.linspace(0, 0.5, 6)
    results = {name: [] for name in noise_funcs}

    # 5. Оценка точности при разных видах шума
    for name, func in noise_funcs.items():
        t_start = time.time()
        for level in noise_levels:
            preds = []
            for x in testP:
                # биполяризация входа
                inp = torch.where(x > 0.5, torch.tensor(1., device=device), torch.tensor(-1., device=device))
                # добавление шума
                noisy = func(inp, level)
                # восстановление
                out = retrieve(noisy, W, max_iter=args.max_iter)
                # классификация ближайшим прототипом
                sims = (P @ out)
                j = torch.argmax(sims).item()
                preds.append(proto_lbls[j].item())
            acc = np.mean((np.array(preds) == testL.cpu().numpy()).astype(float))
            results[name].append(acc)
        print(f'[Eval {name}] {time.time() - t_start:.2f}s')

    # 6. Текстовый вывод точностей для каждого шума и модели
    print('Точности при разных видах шума:')
    for name, accs in results.items():
        acc_str = ', '.join(f'{level * 100:.0f}%:{acc * 100:.2f}%' for level, acc in zip(noise_levels, accs))
        print(f'{name}: {acc_str}')

    # 7. Построение графика точности по шуму
    plt.figure(figsize=(10, 6))
    for name, accs in results.items():
        plt.plot(noise_levels, accs, marker='o', label=name)
    plt.xlabel('Уровень шума')
    plt.ylabel('Точность')
    plt.title('Робастность сети Хопфилда при разных шумовых моделях для геометрических фигур')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accuracy_noise_chart.png')
    plt.show()

    # 8. Визуализация примеров восстановления для разных типов шума
    n = min(args.display_samples, len(testL))
    noise_level = 0.1

    for noise_name, noise_func in noise_funcs.items():
        plt.figure(figsize=(15, 15))
        plt.suptitle(f'Восстановление с использованием шума {noise_name}, уровень {noise_level}', fontsize=16)

        for i in range(n):
            orig = testP[i].cpu().view(args.img_size, args.img_size).numpy()
            inp = torch.where(testP[i] > 0.5, torch.tensor(1., device=device), torch.tensor(-1., device=device))

            # шум конкретного типа
            noisy = noise_func(inp, noise_level).cpu().view(args.img_size, args.img_size).numpy()

            # восстановление
            rec = retrieve(noise_func(inp, noise_level), W, max_iter=args.max_iter).cpu().view(args.img_size,
                                                                                               args.img_size).numpy()

            # Получаем имя класса
            class_name = train_dataset.classes[testL[i].item()]

            ax = plt.subplot(n, 3, 3 * i + 1)
            ax.imshow(orig, cmap='gray')
            ax.set_title(f'Оригинал ({class_name})')
            ax.axis('off')

            ax = plt.subplot(n, 3, 3 * i + 2)
            ax.imshow(noisy, cmap='gray')
            ax.set_title(f'Шум {int(noise_level * 100)}%')
            ax.axis('off')

            ax = plt.subplot(n, 3, 3 * i + 3)
            ax.imshow(rec, cmap='gray')

            # Определяем предсказанный класс
            inp_noisy = noise_func(inp, noise_level)
            out = retrieve(inp_noisy, W, max_iter=args.max_iter)
            sims = (P @ out)
            pred_idx = torch.argmax(sims).item()
            pred_class = proto_lbls[pred_idx].item()
            pred_class_name = train_dataset.classes[pred_class]

            ax.set_title(f'Восстановлено ({pred_class_name})')
            ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(f'restoration_{noise_name}.png')
        plt.show()

    print(f'[Всего времени] {time.time() - t_global:.2f}s')


if __name__ == '__main__':
    main()
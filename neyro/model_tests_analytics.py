import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import time
import os
from tqdm import tqdm

# Создаем директорию для результатов
os.makedirs('results', exist_ok=True)

# Используем GPU, если он доступен
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Базовые гиперпараметры
IMG_SIZE = 64
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 0.001

# Аугментации + нормализация
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Загружаем данные
train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
test_dataset = datasets.ImageFolder(root='dataset/test', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Получаем названия классов
class_names = train_dataset.classes
print(f"Классы: {class_names}")


# Функция для выбора активации
def get_activation_function(name):
    activations = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(0.01)
    }
    return activations.get(name, nn.ReLU())  # По умолчанию ReLU


# Гибкая нейросеть с изменяемыми слоями и активацией
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


# Функция обучения с расширенным возвратом данных
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    model.train()
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_times': []
    }

    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        # Обучение
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{num_epochs} [Обучение]")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

            # Обновляем progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Валидация
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, return_metrics=True)

        # Сохраняем метрики
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(time.time() - start_time)

        print(f"Эпоха {epoch + 1}/{num_epochs}, Потери: {train_loss:.4f}, Точность: {train_acc:.2f}%, "
              f"Тест потери: {test_loss:.4f}, Тест точность: {test_acc:.2f}%")

        # Снижение LR при отсутствии улучшений
        optimizer.param_groups[0]['lr'] *= 0.95  # Простое экспоненциальное снижение LR

    return history


# Функция оценки с расширенными метриками
def evaluate_model(model, test_loader, criterion=None, return_metrics=False, return_predictions=False):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)

            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if return_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total

    if return_metrics and not return_predictions:
        test_loss = running_loss / len(test_loader) if criterion is not None else 0
        return test_loss, accuracy
    elif return_predictions:
        return np.array(all_preds), np.array(all_labels), accuracy
    else:
        print(f"Точность на тесте: {accuracy:.2f}%")
        return accuracy


# Функция для создания и визуализации матрицы ошибок
def plot_confusion_matrix(model, test_loader, class_names):
    predictions, true_labels, accuracy = evaluate_model(model, test_loader, return_predictions=True)

    # Создаем матрицу ошибок
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Матрица ошибок (Точность: {accuracy:.2f}%)')
    plt.ylabel('Правильный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.show()

    # Выводим подробный отчет о классификации
    report = classification_report(true_labels, predictions, target_names=class_names)
    print("Отчет о классификации:")
    print(report)

    return predictions, true_labels


# Функция для анализа влияния изменяемых параметров
def experiment_parameters():
    input_size = IMG_SIZE * IMG_SIZE
    hidden_size = 1024
    num_classes = len(train_loader.dataset.classes)

    # Параметры для экспериментов
    layers_to_test = [1, 2, 3, 4, 5]
    activations_to_test = ['relu', 'sigmoid', 'tanh', 'leaky_relu']

    # Сохраняем результаты экспериментов
    results = []

    for num_layers in layers_to_test:
        for activation in activations_to_test:
            print(f"\n### Эксперимент: слоев={num_layers}, активация={activation} ###")

            # Создаем модель
            model = FlexibleFFNN(input_size, hidden_size, num_classes, num_layers, activation)
            model.to(device)

            # Функция потерь и оптимизатор
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

            # Обучаем модель
            history = train_model(model, train_loader, test_loader, criterion, optimizer,
                                  num_epochs=5)  # Меньше эпох для экспериментов

            # Оцениваем модель
            preds, labels, accuracy = evaluate_model(model, test_loader, return_predictions=True)

            # Вычисляем дополнительные метрики
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

            # Сохраняем результаты
            results.append({
                'num_layers': num_layers,
                'activation': activation,
                'final_train_loss': history['train_loss'][-1],
                'final_train_acc': history['train_acc'][-1],
                'test_acc': accuracy,
                'precision': precision * 100,
                'recall': recall * 100,
                'f1': f1 * 100,
                'avg_epoch_time': np.mean(history['epoch_times'])
            })

            # Сохраняем модель
            torch.save(model.state_dict(), f'results/model_layers{num_layers}_{activation}.pth')

            # Визуализируем процесс обучения
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['test_loss'], label='Test Loss')
            plt.title(f'Потери (L={num_layers}, A={activation})')
            plt.xlabel('Эпоха')
            plt.ylabel('Потери')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='Train Accuracy')
            plt.plot(history['test_acc'], label='Test Accuracy')
            plt.title(f'Точность (L={num_layers}, A={activation})')
            plt.xlabel('Эпоха')
            plt.ylabel('Точность (%)')
            plt.legend()

            plt.tight_layout()
            plt.savefig(f'results/training_plot_L{num_layers}_A{activation}.png')
            plt.close()

    # Преобразуем результаты в DataFrame для анализа
    results_df = pd.DataFrame(results)
    print("\nРезультаты экспериментов:")
    print(results_df)

    # Сохраняем результаты в CSV
    results_df.to_csv('results/experiment_results.csv', index=False)

    return results_df


# Функция для визуализации результатов экспериментов
def visualize_experiment_results(results_df):
    # График зависимости точности от количества слоев для разных функций активации
    plt.figure(figsize=(12, 8))

    # Точность
    plt.subplot(2, 2, 1)
    for activation in results_df['activation'].unique():
        df_subset = results_df[results_df['activation'] == activation]
        plt.plot(df_subset['num_layers'], df_subset['test_acc'], marker='o', label=activation)
    plt.title('Зависимость точности от числа слоев')
    plt.xlabel('Число скрытых слоев')
    plt.ylabel('Точность (%)')
    plt.legend()
    plt.grid(True)

    # F1-мера
    plt.subplot(2, 2, 2)
    for activation in results_df['activation'].unique():
        df_subset = results_df[results_df['activation'] == activation]
        plt.plot(df_subset['num_layers'], df_subset['f1'], marker='o', label=activation)
    plt.title('Зависимость F1-меры от числа слоев')
    plt.xlabel('Число скрытых слоев')
    plt.ylabel('F1-мера (%)')
    plt.legend()
    plt.grid(True)

    # Время обучения
    plt.subplot(2, 2, 3)
    for activation in results_df['activation'].unique():
        df_subset = results_df[results_df['activation'] == activation]
        plt.plot(df_subset['num_layers'], df_subset['avg_epoch_time'], marker='o', label=activation)
    plt.title('Зависимость времени обучения от числа слоев')
    plt.xlabel('Число скрытых слоев')
    plt.ylabel('Среднее время эпохи (сек)')
    plt.legend()
    plt.grid(True)

    # Потери на обучении
    plt.subplot(2, 2, 4)
    for activation in results_df['activation'].unique():
        df_subset = results_df[results_df['activation'] == activation]
        plt.plot(df_subset['num_layers'], df_subset['final_train_loss'], marker='o', label=activation)
    plt.title('Зависимость потерь от числа слоев')
    plt.xlabel('Число скрытых слоев')
    plt.ylabel('Потери на обучении')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/parameter_comparison.png')
    plt.show()

    # Сравнение функций активации по эффективности
    plt.figure(figsize=(10, 6))
    activation_metrics = results_df.groupby('activation').mean()
    metrics = ['test_acc', 'precision', 'recall', 'f1']
    activation_metrics[metrics].plot(kind='bar', figsize=(12, 6))
    plt.title('Сравнение функций активации по эффективности')
    plt.ylabel('Значение (%)')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('results/activation_comparison.png')
    plt.show()

    # Тепловая карта для визуализации всех параметров
    plt.figure(figsize=(14, 8))
    pivot_acc = results_df.pivot(index='activation', columns='num_layers', values='test_acc')
    sns.heatmap(pivot_acc, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Тепловая карта точности по слоям и функциям активации')
    plt.xlabel('Число слоев')
    plt.ylabel('Функция активации')
    plt.tight_layout()
    plt.savefig('results/accuracy_heatmap.png')
    plt.show()


# Функция для анализа ошибочных классификаций
def analyze_misclassifications(model, test_loader, class_names):
    model.eval()
    errors = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            flattened_inputs = inputs.view(inputs.size(0), -1)
            outputs = model(flattened_inputs)
            _, predicted = torch.max(outputs, 1)

            # Находим индексы ошибочных предсказаний
            mask = (predicted != labels)
            if mask.any():
                error_indices = mask.nonzero(as_tuple=True)[0]
                for idx in error_indices:
                    errors.append({
                        'image': inputs[idx].cpu(),
                        'true': class_names[labels[idx].item()],
                        'predicted': class_names[predicted[idx].item()],
                        'confidence': torch.softmax(outputs[idx], dim=0)[predicted[idx]].item()
                    })

    # Вывод статистики ошибок
    error_count = len(errors)
    total_samples = len(test_loader.dataset)
    print(f"Всего ошибок: {error_count} из {total_samples} ({error_count / total_samples * 100:.2f}%)")

    # Анализ типов ошибок
    error_types = {}
    for error in errors:
        key = f"{error['true']} → {error['predicted']}"
        if key in error_types:
            error_types[key] += 1
        else:
            error_types[key] = 1

    # Сортировка ошибок по частоте
    sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)

    # Визуализация наиболее частых ошибок
    plt.figure(figsize=(10, 6))
    error_labels = [item[0] for item in sorted_errors[:10]]  # топ-10 ошибок
    error_values = [item[1] for item in sorted_errors[:10]]
    plt.bar(error_labels, error_values)
    plt.xlabel('Тип ошибки')
    plt.ylabel('Количество')
    plt.title('Наиболее частые типы ошибок классификации')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/error_types.png')
    plt.show()

    # Визуализация примеров ошибочных предсказаний
    if len(errors) > 0:
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < min(9, len(errors)):
                error = errors[i]
                img = error['image'].permute(1, 2, 0).cpu().numpy()
                img = (img * 0.5) + 0.5  # Денормализация
                ax.imshow(img.squeeze(), cmap='gray')
                ax.set_title(
                    f"Истинный: {error['true']}\nПредсказан: {error['predicted']}\nУверен.: {error['confidence']:.2f}")
                ax.axis('off')
        plt.tight_layout()
        plt.savefig('results/error_examples.png')
        plt.show()

    return errors, error_types


# Главная функция для запуска экспериментов
def main():
    print("Запуск экспериментов с параметрами нейронной сети...")
    results_df = experiment_parameters()

    print("\nВизуализация результатов экспериментов...")
    visualize_experiment_results(results_df)

    # Находим лучшую конфигурацию
    best_config = results_df.loc[results_df['test_acc'].idxmax()]
    print(f"\nЛучшая конфигурация: {best_config.to_dict()}")

    # Создаем и обучаем лучшую модель
    input_size = IMG_SIZE * IMG_SIZE
    hidden_size = 1024
    num_classes = len(train_loader.dataset.classes)

    best_model = FlexibleFFNN(
        input_size,
        hidden_size,
        num_classes,
        int(best_config['num_layers']),
        best_config['activation']
    )
    best_model.to(device)

    # Загружаем сохраненные веса лучшей модели
    best_model_path = f"results/model_layers{int(best_config['num_layers'])}_{best_config['activation']}.pth"
    best_model.load_state_dict(torch.load(best_model_path))

    # Создаем матрицу ошибок для лучшей модели
    print("\nАнализ матрицы ошибок для лучшей модели...")
    predictions, true_labels = plot_confusion_matrix(best_model, test_loader, class_names)

    # Анализируем ошибочные классификации
    print("\nАнализ ошибочных классификаций...")
    errors, error_types = analyze_misclassifications(best_model, test_loader, class_names)

    print("\nВсе эксперименты завершены. Результаты сохранены в директории 'results/'")


if __name__ == "__main__":
    main()
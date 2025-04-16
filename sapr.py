import numpy as np
import matplotlib.pyplot as plt
import os # Для создания папки

# --- Параметры ---
k = 2
m = 1
# Диапазон изменения бета (нагрузки на один "исходный" сервер)
beta_values = np.linspace(0.01, 0.99, 200) # От 0.01 до 0.99

# --- Функции для расчета показателей (остаются без изменений) ---

def calculate_metrics_a(beta):
    """ Расчет для структуры 'a' (M/M/1/2, N=2, rho=beta) """
    p_otk = np.zeros_like(beta)
    k_z = np.zeros_like(beta)
    n_o = np.zeros_like(beta)
    j = np.zeros_like(beta)
    t_ozh = np.zeros_like(beta)
    t_s = np.zeros_like(beta)
    non_zero_beta = beta != 0
    b = beta[non_zero_beta]
    denom = (1 - b**3)
    denom[denom == 0] = 1e-10
    p0 = (1 - b) / denom
    p1 = p0 * b
    p2 = p0 * b**2
    p_otk[non_zero_beta] = p2
    k_z[non_zero_beta] = 1 - p0
    n_o[non_zero_beta] = p2
    j[non_zero_beta] = 1*p1 + 2*p2
    t_ozh[non_zero_beta] = b / (1 + b)
    t_s[non_zero_beta] = (1 + 2*b) / (1 + b)
    t_s[beta == 0] = 1
    return p_otk, k_z, n_o, j, t_ozh, t_s

def calculate_metrics_c(beta):
    """ Расчет для структуры 'c' (M/M/1/2, N=2, rho=beta, но Lambda=k*lambda, Mu=k*mu) """
    p_otk, k_z, n_o, j, _, _ = calculate_metrics_a(beta)
    t_ozh = np.zeros_like(beta)
    t_s = np.zeros_like(beta)
    non_zero_beta = beta != 0
    b = beta[non_zero_beta]
    t_ozh[non_zero_beta] = (b / (1 + b)) / k
    t_s[non_zero_beta] = t_ozh[non_zero_beta] + 0.5
    t_s[beta == 0] = 0.5
    return p_otk, k_z, n_o, j, t_ozh, t_s

def calculate_metrics_b(beta):
    """ Расчет для структуры 'b' (M/M/k/k+m = M/M/2/3, N=3, rho=beta, a=k*rho=2*beta) """
    p_otk = np.zeros_like(beta)
    k_z = np.zeros_like(beta)
    n_o = np.zeros_like(beta)
    j = np.zeros_like(beta)
    t_ozh = np.zeros_like(beta)
    t_s = np.zeros_like(beta)
    non_zero_beta = beta != 0
    b = beta[non_zero_beta]
    a = k * b
    p0_denom = (1 + a) + (a**2 / 2) * (1 + b)
    p0 = 1 / p0_denom
    p1 = a * p0
    p2 = (a**2 / 2) * p0
    p3 = p2 * b
    p_otk[non_zero_beta] = p3
    j_s = 1*p1 + 2*p2 + 2*p3
    k_z[non_zero_beta] = j_s / k
    n_o[non_zero_beta] = p3
    j[non_zero_beta] = j_s + n_o[non_zero_beta]
    lambda_eff_norm = k * b * (1 - p_otk[non_zero_beta])
    lambda_eff_norm[lambda_eff_norm == 0] = 1e-10
    t_ozh[non_zero_beta] = n_o[non_zero_beta] / lambda_eff_norm
    t_s[non_zero_beta] = t_ozh[non_zero_beta] + 1
    t_s[beta == 0] = 1
    return p_otk, k_z, n_o, j, t_ozh, t_s

# --- Расчет значений ---
results_a = calculate_metrics_a(beta_values)
results_b = calculate_metrics_b(beta_values)
results_c = calculate_metrics_c(beta_values)

metrics_data = {
    'a': results_a,
    'b': results_b,
    'c': results_c,
}

metric_names_full = [
    r'$P_{отк}$ (Вероятность отказа)',
    r'$K_з$ (Коэф. загрузки)',
    r'$\bar{n}_о$ (Ср. число в очереди)',
    r'$\bar{j}$ (Ср. число в системе)',
    r'$\bar{t}_{ож} \cdot \mu$ (Норм. ср. время ожидания)',
    r'$\bar{t}_с \cdot \mu$ (Норм. ср. время в системе)'
]
# Короткие имена для файлов
metric_names_short = [
    'P_otk', 'K_z', 'n_o', 'j', 'T_ozh_norm', 'T_s_norm'
]

# --- Создание папки для графиков (если ее нет) ---
output_dir = "smo_graphs_variant5"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Создана папка: {output_dir}")

# --- Построение и сохранение отдельных графиков ---
plt.style.use('seaborn-v0_8-whitegrid')

for i, (metric_name_full, metric_name_short) in enumerate(zip(metric_names_full, metric_names_short)):
    # Создаем НОВУЮ фигуру для каждого графика
    fig, ax = plt.subplots(figsize=(8, 6)) # Задаем размер фигуры

    # Рисуем данные для текущего показателя
    ax.plot(beta_values, metrics_data['a'][i], label='Структура а (M/M/1/2) x2', linestyle='-', color='blue')
    ax.plot(beta_values, metrics_data['b'][i], label='Структура б (M/M/2/3)', linestyle='--', color='red')
    ax.plot(beta_values, metrics_data['c'][i], label='Структура в (M/M/1/2) fast', linestyle=':', color='green')

    # Настройки графика
    ax.set_title(metric_name_full)
    ax.set_xlabel(r'$\beta = \lambda / \mu$ (Нагрузка на 1 сервер/поток)')
    ax.set_ylabel('Значение показателя')
    ax.legend()
    ax.grid(True)

    # Установка пределов по Y
    if i == 0 or i == 1: # P_отк или K_з
        ax.set_ylim(0, 1.05)
    else:
        ax.set_ylim(bottom=0) # Остальные неотрицательны

    # Сохранение графика в файл
    file_path = os.path.join(output_dir, f"graph_{i+1}_{metric_name_short}.png")
    plt.savefig(file_path, dpi=150) # Сохраняем с хорошим разрешением
    print(f"График сохранен: {file_path}")

    # Закрываем текущую фигуру, чтобы освободить память
    plt.close(fig)

print("\nПостроение и сохранение всех 6 графиков завершено.")
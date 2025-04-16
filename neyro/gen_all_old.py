import cv2
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt

# Размер изображения
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
    x1 = random.randint(10, 60)
    x2 = x1 + random.randint(10, 25)
    y1 = random.randint(MARGIN, IMG_SIZE // 2)
    y2 = y1 + random.randint(20, 40)
    x3, x4 = x1 - random.randint(5, 15), x2 + random.randint(5, 15)
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

# Генерация и отображение фигур
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, (shape_name, shape_func) in enumerate(shapes.items()):
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    shape_func(img)
    ax = axes[i // 5, i % 5]
    ax.imshow(img, cmap="gray")
    ax.set_title(shape_name)
    ax.axis("off")

plt.tight_layout()
plt.show()
## ChernabogPy: Визуализация гравитационных искажений черных дыр

[black-hole.webm](https://github.com/viktor-akusoff/ChernabogPy/assets/124511385/76787833-fdb5-43eb-a682-10830be4ce34)

ChernabogPy - программа на Python, предназначенная для визуализации гравитационных искажений, вызываемых черными дырами, с использованием алгоритма нелинейной трассировки лучей. Программа позволяет исследовать эффекты искривления пространства-времени, описанные метриками Шварцшильда и Рейснера-Нордстрёма, и создавать реалистичные изображения черных дыр.

Магистерская диссертация связанная с этой программой:
[Некоторые математические модели релятивитских объектов вселенной.pdf](https://github.com/user-attachments/files/15948451/default.pdf)

### Возможности:

* Трассировка лучей с учетом гравитационного поля черных дыр:
    * Метрика Шварцшильда (невращающаяся, незаряженная черная дыра)
    * Метрика Рейснера-Нордстрёма (невращающаяся, заряженная черная дыра)
* Настройка параметров черной дыры (масса, заряд)
* Настройка параметров сцены (камера, фон, объекты)
* Поддержка различных текстур:
    * Однотонные цвета
    * Шахматные узоры
    * Загрузка текстур из файлов изображений
* Визуализация траекторий лучей в 3D
* Сохранение результирующего изображения

### Использование:


1. **Настройка параметров:**

   Откройте файл `main.py` и настройте следующие параметры:

   * **Разрешение изображения:**

     ```python
     w = 640  # Ширина изображения
     h = 320  # Высота изображения
     ```

   * **Текстуры:**

     ```python
     black_hole_texture = CheckersTexture((0, 0, 0.9), (0.5, 0.5, 0), 15)  # Текстура черной дыры
     background_texture = ImageTexture('img/space2.jpg')  # Текстура фона
     ring_texture = CheckersTexture((0.9, 0, 0), (0, 0.5, 0.5), 20)  # Текстура кольца
     ```

   * **Объекты сцены:**

     ```python
     black_hole = Sphere(t([0, 0, 0]), 0.5)  # Черная дыра (позиция, радиус)
     background = Sphere(t([0, 0, 0]), 9, inverse=True)  # Фон (позиция, радиус, инвертировать)
     ring = FlatRing(t([0, 0, 0]), (0.5, 5), t([0, 0, 1]))  # Кольцо (позиция, (внутренний радиус, внешний радиус), нормаль)
     ```

   * **Камера:**

     ```python
     camera = Camera(t([-7, 0, 1.2]), t([0, 0, 0]), (w, h), 1, np.pi/3)
     # Позиция, направление, разрешение, расстояние до экрана, угол обзора
     ```

   * **Стратегия трассировки лучей:**

     ```python
     raytracer = RayTracer(
         scene,
         ReissnerNordstromRayTrace(1, 0.5),  # Выбор метрики (Шварцшильда или Рейснера-Нордстрёма)
         eps=1e-3  # Точность трассировки
     )
     ```

2. **Запуск программы:**

   Запустите файл `start.bat`, чтобы выполнить рендеринг сцены. Программа отобразит индикатор выполнения и сохранит результирующее изображение в файл `blackholeHD.png`.

### Примеры:

* **Линейная трассировка лучей:**

   ```python
   raytracer = RayTracer(scene, LinearRayTrace(), eps=1e-3)
   ```

* **Трассировка лучей в метрике Шварцшильда:**

   ```python
   raytracer = RayTracer(scene, SchwarzschildRayTrace(curvature=3), eps=1e-3)
   ```

* **Трассировка лучей в метрике Рейснера-Нордстрёма:**

   ```python
   raytracer = RayTracer(scene, ReissnerNordstromRayTrace(e=1.5, charge=0.5), eps=1e-3)
   ```

### Дополнительные возможности:

* **Отображение 3D-графика точек пересечения лучей:**

   ```python
   raytracer.view_3d_rays_hits((-15, 15), (-15, 15), (-15, 15), 64, 32)
   ```

* **Визуализация сцены в 3D:**

   ```python
   scene.view_3d_scene((-15, 15), (-15, 15), (-15, 15))
   ```

### Замечания:

* Для ускорения рендеринга рекомендуется использовать GPU. Убедитесь, что у вас установлен PyTorch с поддержкой CUDA.
* Настройка параметров сцены и трассировки лучей может потребовать некоторого экспериментального подбора.
* Исходный код программы доступен в папке `chernabog`.

### Лицензия:

MIT License

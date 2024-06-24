## ChernabogPy: Visualization of Gravitational Distortions by Black Holes

[black-hole.webm](https://github.com/viktor-akusoff/ChernabogPy/assets/124511385/76787833-fdb5-43eb-a682-10830be4ce34)

ChernabogPy is a Python program designed to visualize gravitational distortions caused by black holes using a nonlinear ray tracing algorithm. The program allows you to explore the effects of spacetime curvature described by the Schwarzschild and Reissner-Nordström metrics and create realistic images of black holes.

This program was created as part of a [master's thesis](https://github.com/user-attachments/files/15948451/default.pdf).

### Features:

* Ray tracing considering the gravitational field of black holes:
    * Schwarzschild metric (non-rotating, uncharged black hole)
    * Reissner-Nordström metric (non-rotating, charged black hole)
* Customization of black hole parameters (mass, charge)
* Customization of scene parameters (camera, background, objects)
* Support for various textures:
    * Solid colors
    * Checkerboard patterns
    * Loading textures from image files
* Visualization of ray trajectories in 3D
* Saving the resulting image

### Usage:

1. **Parameter Configuration:**

   Open the `main.py` file and configure the following parameters:

   * **Image Resolution:**

     ```python
     w = 640  # Image width
     h = 320  # Image height
     ```

   * **Textures:**

     ```python
     black_hole_texture = CheckersTexture((0, 0, 0.9), (0.5, 0.5, 0), 15)  # Black hole texture
     background_texture = ImageTexture('img/space2.jpg')  # Background texture
     ring_texture = CheckersTexture((0.9, 0, 0), (0, 0.5, 0.5), 20)  # Ring texture
     ```

   * **Scene Objects:**

     ```python
     black_hole = Sphere(t([0, 0, 0]), 0.5)  # Black hole (position, radius)
     background = Sphere(t([0, 0, 0]), 9, inverse=True)  # Background (position, radius, invert)
     ring = FlatRing(t([0, 0, 0]), (0.5, 5), t([0, 0, 1]))  # Ring (position, (inner radius, outer radius), normal)
     ```

   * **Camera:**

     ```python
     camera = Camera(t([-7, 0, 1.2]), t([0, 0, 0]), (w, h), 1, np.pi/3)
     # Position, direction, resolution, distance to screen, field of view
     ```

   * **Ray Tracing Strategy:**

     ```python
     raytracer = RayTracer(
         scene,
         ReissnerNordstromRayTrace(1, 0.5),  # Choose metric (Schwarzschild or Reissner-Nordström)
         eps=1e-3  # Tracing precision
     )
     ```

2. **Run the Program:**

   Run the `start.bat` file to render the scene. The program will display a progress bar and save the resulting image to the `blackholeHD.png` file.

### Examples:

* **Linear Ray Tracing:**

   ```python
   raytracer = RayTracer(scene, LinearRayTrace(), eps=1e-3)
   ```

* **Ray Tracing with the Schwarzschild Metric:**

   ```python
   raytracer = RayTracer(scene, SchwarzschildRayTrace(curvature=3), eps=1e-3)
   ```

* **Ray Tracing with the Reissner-Nordström Metric:**

   ```python
   raytracer = RayTracer(scene, ReissnerNordstromRayTrace(e=1.5, charge=0.5), eps=1e-3)
   ```

### Additional Features:

* **Display 3D Graph of Ray Intersection Points:**

   ```python
   raytracer.view_3d_rays_hits((-15, 15), (-15, 15), (-15, 15), 64, 32)
   ```

* **Visualize the Scene in 3D:**

   ```python
   scene.view_3d_scene((-15, 15), (-15, 15), (-15, 15))
   ```

### Notes:

* It is recommended to use a GPU for faster rendering. Make sure you have PyTorch installed with CUDA support.
* Configuring scene parameters and ray tracing settings may require some experimental adjustments.
* The program's source code is available in the `chernabog` folder.

### License:

MIT License

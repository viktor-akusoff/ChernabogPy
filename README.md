## ChernabogPy: Visualization of Gravitational Lensing by Black Holes

[black-hole.webm](https://github.com/viktor-akusoff/ChernabogPy/assets/124511385/76787833-fdb5-43eb-a682-10830be4ce34)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ChernabogPy is a Python program designed to visualize gravitational lensing effects caused by black holes using a nonlinear ray tracing algorithm. The program allows you to explore the effects of spacetime curvature described by the Schwarzschild and Reissner-Nordström metrics and create realistic images of black holes.

This program was created as part of a [master's thesis](https://github.com/user-attachments/files/15948451/default.pdf).

### Features:

* Ray tracing considering the gravitational field of black holes:
    * Schwarzschild metric (non-rotating, uncharged black hole)
    * Reissner-Nordström metric (non-rotating, charged black hole)
* Adjustment of black hole parameters (mass, charge)
* Scene parameter customization (camera, background, objects)
* Support for various textures:
    * Solid colors
    * Checkerboard patterns
    * Loading textures from image files
* Visualization of ray trajectories in 3D
* Saving the rendered image
* Loading the scene from an XML file
* **GPU/CPU rendering** — select compute device via `--device` flag

### Usage:

ChernabogPy uses XML files to describe the scene. The program accepts the following arguments:

```
python main.py [input.xml] [output.png] [--device auto|cuda|cpu]
```

| Argument | Default | Description |
|---|---|---|
| `input.xml` | `scene.xml` | Path to scene XML file |
| `output.png` | `output.png` | Path to output PNG file |
| `--device` | `auto` | Compute device: `auto` (detect GPU), `cuda` (force GPU), `cpu` (force CPU) |

Usage examples:

```bash
# Auto-detect GPU (default)
python main.py scene.xml output.png

# Force GPU rendering
python main.py scene.xml output.png --device cuda

# Force CPU rendering (no GPU required)
python main.py scene.xml output.png --device cpu

# Use bundled test scene
python main.py scene_test.xml output.png --device cuda
```

If arguments are not specified, the program will use the default files `scene.xml` and `output.png`.

#### XML File Structure:

```xml
<scene>
    <camera>
        <position x="-7" y="0" z="1.2"/> 
        <look_at x="0" y="0" z="0"/> 
        <resolution width="640" height="320"/>
        <distance_to_screen>1</distance_to_screen>
        <fov>1.0472</fov> 
    </camera>
    <objects>
        <object type="sphere">
            <position x="0" y="0" z="0"/>
            <radius>0.5</radius>
            <texture type="color" r="0" g="0" b="0"/>
        </object>
        <object type="sphere">
            <position x="0" y="0" z="0"/>
            <radius>9</radius>
            <inverse>true</inverse> 
            <texture type="checkers">
                <color1 r="0" g="0.2" b="0.7"/>
                <color2 r="0.3" g="0.9" b="0"/>
                <length>25</length>
            </texture>
        </object>
        <object type="flat_ring">
            <position x="0" y="0" z="0"/>
            <inner_radius>0.5</inner_radius>
            <outer_radius>5</outer_radius>
            <norm_vector x="0" y="0" z="1"/>
            <texture type="image">
                <address>./img/rings.jpg</address>
            </texture>
        </object>
    </objects>
    <raytracer>
        <type>ReissnerNordstromRayTrace</type>
        <e>1</e>
        <charge>0.5</charge>
        <eps>1e-3</eps> 
    </raytracer>
</scene>
```

Element Description:

* **`<scene>`:** Root element.
    * **`<camera>`:** Camera parameters.
        * **`<position>`:** Camera position (x, y, z).
        * **`<look_at>`:** Point the camera is looking at (x, y, z).
        * **`<resolution>`:** Image resolution (width, height).
        * **`<distance_to_screen>`:** Distance from the camera to the screen.
        * **`<fov>`:** Camera's field of view (in radians).
    * **`<objects>`:** List of scene objects.
        * **`<object>`:** Object description.
            * **`type`:** Object type (sphere, flat_ring).
            * **`<position>`:** Object position (x, y, z).
            * **`<radius>`:** Sphere radius.
            * **`<inverse>`:** Invert the sphere (true/false), used for the background.
            * **`<inner_radius>`:** Inner radius of the ring.
            * **`<outer_radius>`:** Outer radius of the ring.
            * **`<norm_vector>`:** Normal vector of the ring (x, y, z).
            * **`<texture>`:** Object texture.
                * **`type`:** Texture type (color, checkers, image).
                * **`r`, `g`, `b`:** Color values for solid color texture (0.0 - 1.0). 
                * **`<color1>`:** First color for the checkerboard texture (r, g, b).
                * **`<color2>`:** Second color for the checkerboard texture (r, g, b).
                * **`<length>`:** Cell size for the checkerboard texture.
                * **`<address>`:** Path to the image file for the texture.
    * **`<raytracer>`:** Ray tracing parameters.
        * **`<type>`:**  Ray tracer type (LinearRayTrace, SchwarzschildRayTrace, ReissnerNordstromRayTrace).
        * **`<curvature>`:** Curvature for the Schwarzschild metric.
        * **`<e>`:** `e` parameter for the Reissner-Nordström metric.
        * **`<charge>`:** Charge for the Reissner-Nordström metric.
        * **`<eps>`:** Ray tracing precision.

### Creating an Executable:

1. Install PyInstaller:

   ```bash
   pip install pyinstaller
   ```

2. Build the executable:

   ```bash
   pyinstaller --onefile main.py 
   ```

The executable file will be located in the `dist` folder.

### Notes:

* For faster rendering, using a GPU is recommended. Ensure you have PyTorch installed with CUDA support.
* Without a GPU, rendering a 1024×768 scene may take 15–30 minutes. Use `--device cpu` explicitly to avoid errors on machines without CUDA.
* Fine-tuning scene and ray tracing parameters might require some experimentation.
* A ready-to-use test scene with procedural checkerboard textures is included as `scene_test.xml`.
* The program's source code is available in the `chernabog` folder.

### License:

MIT License

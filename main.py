import torch
import numpy as np
import asyncio
import xml.etree.ElementTree as ET
import sys
from chernabog.misc import TensorFabric
from chernabog.entities import Sphere, FlatRing, Scene, Camera, DecoratedPair
from chernabog.textures import ColorTexture, CheckersTexture, ImageTexture
from chernabog.render import LinearRayTrace, RayTracer, SchwarzschildRayTrace, ReissnerNordstromRayTrace
import matplotlib.pyplot as plt

def load_texture_from_xml(texture_node):
    texture_type = texture_node.get('type')
    if texture_type == 'color':
        color = ColorTexture(
            (
                float(texture_node.get('r')),
                float(texture_node.get('g')),
                float(texture_node.get('b'))
            )
        )
        return color
    elif texture_type == 'checkers':
        color1 = (
            float(texture_node.find('color1').get('r')),
            float(texture_node.find('color1').get('g')),
            float(texture_node.find('color1').get('b'))
        )
        color2 = (
            float(texture_node.find('color2').get('r')),
            float(texture_node.find('color2').get('g')),
            float(texture_node.find('color2').get('b'))
        )
        length = float(texture_node.find('length').text)
        return CheckersTexture(color1, color2, length)
    elif texture_type == "image":
        address = texture_node.find('address').text
        return ImageTexture(address)
    else:
        raise ValueError(f"Unsupported texture type: {texture_type}")

def load_object_from_xml(object_node, t):
    object_type = object_node.get('type')
    position = t([
        float(object_node.find('position').get('x')),
        float(object_node.find('position').get('y')),
        float(object_node.find('position').get('z'))
    ])

    if object_type == 'sphere':
        radius = float(object_node.find('radius').text)
        inverse = object_node.find('inverse') is not None and object_node.find('inverse').text.lower() == 'true'
        entity = Sphere(position, radius, inverse)
    elif object_type == 'flat_ring':
        inner_radius = float(object_node.find('inner_radius').text)
        outer_radius = float(object_node.find('outer_radius').text)
        norm_vector = t([
            float(object_node.find('norm_vector').get('x')),
            float(object_node.find('norm_vector').get('y')),
            float(object_node.find('norm_vector').get('z'))
        ])
        entity = FlatRing(position, (inner_radius, outer_radius), norm_vector)
    else:
        raise ValueError(f"Unsupported object type: {object_type}")

    texture = load_texture_from_xml(object_node.find('texture'))
    return DecoratedPair(entity, texture)


def load_scene_from_xml(filename="scene.xml"):
    tree = ET.parse(filename)
    root = tree.getroot()
    t = TensorFabric(torch.float64)

    # Camera
    camera_node = root.find('camera')
    position = t([
        float(camera_node.find('position').get('x')),
        float(camera_node.find('position').get('y')),
        float(camera_node.find('position').get('z'))
    ])
    look_at = t([
        float(camera_node.find('look_at').get('x')),
        float(camera_node.find('look_at').get('y')),
        float(camera_node.find('look_at').get('z'))
    ])
    resolution = (
        int(camera_node.find('resolution').get('width')),
        int(camera_node.find('resolution').get('height'))
    )
    distance_to_screen = float(camera_node.find('distance_to_screen').text)
    fov = float(camera_node.find('fov').text)

    camera = Camera(position, look_at, resolution, distance_to_screen, fov)
    scene = Scene(camera)

    # Objects
    for object_node in root.find('objects').findall('object'):
        pair = load_object_from_xml(object_node, t)
        scene.push_entity(pair)

    # Raytracer
    raytracer_node = root.find('raytracer')
    raytracer_type = raytracer_node.find('type').text
    eps = float(raytracer_node.find('eps').text)

    if raytracer_type == 'LinearRayTrace':
        strategy = LinearRayTrace()
    elif raytracer_type == 'SchwarzschildRayTrace':
        curvature = float(raytracer_node.find('curvature').text)
        strategy = SchwarzschildRayTrace(curvature)
    elif raytracer_type == 'ReissnerNordstromRayTrace':
        e = float(raytracer_node.find('e').text)
        charge = float(raytracer_node.find('charge').text)
        strategy = ReissnerNordstromRayTrace(e, charge)
    else:
        raise ValueError(f"Unsupported raytracer type: {raytracer_type}")

    raytracer = RayTracer(scene, strategy, eps=eps)
    return raytracer


async def main(raytracer: RayTracer):
    async with asyncio.TaskGroup() as tg:
        tg.create_task(raytracer.async_calc_points())
        tg.create_task(raytracer.async_show_progress())

if __name__ == "__main__":
    input_xml = "scene.xml"
    output_png = "output.png"

    if len(sys.argv) > 1: 
        input_xml = sys.argv[1]
    if len(sys.argv) > 2:
        output_png = sys.argv[2]

    raytracer = load_scene_from_xml(filename=input_xml) 
    asyncio.run(main(raytracer))
    img_data = raytracer.calc_image()
    plt.imsave(output_png, img_data)
    plt.show() 
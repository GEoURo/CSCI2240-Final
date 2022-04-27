import bpy
import random
import math
import numpy as np
import mathutils
import os
import json 

'''
Orients a camera object to look at a particular point in space.
reference: https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
'''
def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()

'''
Uniformly samples the surface of unit hemisphere centered at (0,0,0)
pointing upward in the +Z direction.
'''
def uniformSampleHemisphere():
   z = random.uniform(0,1)
   r = math.sqrt(max(0.0, 1.0 - z ** 2))
   phi = 2 * math.pi * random.uniform(0,1)
   return np.array([r * math.cos(phi), r * math.sin(phi), z])

'''
Samples a new camera location and adjusts the camera orientation
'''
def randomlyMoveCamera(camera_obj):
    # Sample hemisphere point
    loc = uniformSampleHemisphere()
    
    # Move camera to loc
    camera_obj.location = mathutils.Vector(loc)
    
    # Update internal Blender matrices, IMPORTANT!
    bpy.context.view_layer.update()
    
    # Orient camera to look at origin
    look_at(camera_obj, mathutils.Vector((0,0,0)))

'''
Renders variable number of random view. Saves images and transforms to output_dir.
'''
def renderRandomViews(output_dir='./', num_views=2, train_test="train"):
    my_camera = bpy.data.objects["TEST_CAMERA"]    
    print('Camera angle_x: ', bpy.context.scene.camera.data.angle_x)
    output_dict = {
        'camera_angle_x': bpy.context.scene.camera.data.angle_x,
        'frames': []
    }
    
    for i in range(num_views):
        print('Rendering view ', i)
        randomlyMoveCamera(my_camera)
        filename = f'r_{i}.png'
        bpy.context.scene.render.filepath = os.path.join(output_dir, filename)
        bpy.ops.render.render(write_still = True)
        trans_mat = my_camera.matrix_world
        output_dict['frames'].append({
            'file_path': f'./{train_test}/r_{i}',
            'transform_matrix': [
                list(trans_mat[0]),
                list(trans_mat[1]),
                list(trans_mat[2]),
                list(trans_mat[3]),
            ]
        })
    
    json_path = os.path.join(output_dir, f'transforms_{train_test}.json')
    with open(json_path, 'w') as outfile:
        json.dump(output_dict, outfile, indent=2)
    
    
renderRandomViews(
    output_dir='C:\\Users\\Tank\\Documents\\Brown\\Courses\\Graphics\\CSCI2240-Final\\blender_scenes\\test1\\views',
    num_views=10
)

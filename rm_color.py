import open3d as o3d
import argparse
import os
import numpy as np
import os.path as osp
import trimesh

class MyVertexColor(trimesh.visual.color.VertexColor):
    """
    Create a simple visual object to hold just vertex colors
    for objects such as PointClouds.
    """

    def __init__(self, colors=None, obj=None):
        """
        Create a vertex color visual
        """
        super().__init__(colors=None, obj=None)
    
    @property
    def kind(self):
        return 'none'
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default ='/home/dichang/code/cascade_nerf_train_history/normal_depth_loss/fusion/num2')
    parser.add_argument('--target_folder', type=str, default ='/home/dichang/code/cascade_nerf_train_history/normal_depth_loss/fusion/num2_rm_color')
    parser.add_argument('--scene_names', nargs='+', default =None)
    args = parser.parse_args()

    data_folder = args.data_folder
    target_folder = args.target_folder
    
    if not osp.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)

    if args.scene_names is not None:
        scene_names = args.scene_names
    else:
        scene_names = os.listdir(data_folder)
        scene_names = [i for i in scene_names if os.path.splitext(i)[-1] == '.ply']
    
    for scene_name in scene_names:
        source_path = os.path.join(data_folder, scene_name)
        target_path = os.path.join(target_folder, scene_name)
        # source_pt = o3d.io.read_point_cloud(source_path)
        
        print("Remove color of ", scene_name)
        # # target_pt = source_pt.select_down_sample(target_idx)
        # source_pt.colors = source_pt.normals
        # o3d.io.write_point_cloud(target_path, source_pt)
        source_pt = trimesh.load(source_path)
        target_pt = trimesh.PointCloud(vertices=source_pt.vertices, colors=None)
        target_pt.visual = MyVertexColor(colors=None, obj=target_pt)
        target_pt.export(target_path)
        pass
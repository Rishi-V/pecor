import generate_data
import ycb
import table_pose
from pycocotools.coco import COCO
import numpy as np
import pdb
from scipy.spatial.transform import Rotation as R
import time
import os

object_dict = {2:"002_master_chef_can", 3:"003_cracker_box",4:"004_sugar_box", 5: "005_tomato_soup_can",9:"009_gelatin_box",10:"010_potted_meat_can"}
#settings 
search_object =  object_dict[4]
file_names = ["0058"]
#input picture index
IND_list = [
    [77]
]
#### Example paths: Please change all these four for your set up
YCB_dataset_dir = '/home/jessy/perch2.0/data/YCB_Video_Dataset/' #original YCB data folder
output_dir = "/home/jessy/perch-clean/output/" #output folder for accuracy and output pose
data_dir = "/home/jessy/perch-clean/data/" #render image folder that C++ creates, should be same in generate_data/src/color_only/include/color_only/file_paths.h
model_dir = "/home/jessy/Desktop/models/" #model folder, should be same in file_paths.h

for i in range(len(file_names)):
    file_name = file_names[i]
    INDs = IND_list[i]
# for ground truth pose retrieval using COCO since all info in one instances_keyframe_bbox_pose.json for YCB dataset
    example_coco = COCO(YCB_dataset_dir +'instances_keyframe_bbox_pose.json')
    image_ids = example_coco.getImgIds()
    category_id_to_names = example_coco.loadCats(example_coco.getCatIds())
    category_names_to_id = {}
    category_ids = example_coco.getCatIds(catNms=['square', 'shape'])
    for category in category_id_to_names:
        category_names_to_id[category['name']] = category['id']
# open output file
    output_f = open(output_dir+file_name+"/output_"+search_object+".txt", "a")
    error_f = open(output_dir+file_name+"/error_"+search_object+".txt", "a")
# for every image
    for IND in INDs:
        name =  'data/'+file_name+'/'+"{:06d}".format(IND)+"-color.png"
        for i in range(len(image_ids)):
            image_data = example_coco.loadImgs(image_ids[i])[0]
            if image_data['file_name'] == name:
                found = True
                break
        annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], iscrowd=None)
        annotations = example_coco.loadAnns(annotation_ids)
        #find ground truth loc and orientation
        object_loc = None
        object_quat = None
        for annotation in annotations:
            class_name = category_id_to_names[annotation['category_id']]['name']
            if class_name == search_object:
                gt_pose = annotation["location"]+ annotation["quaternion_xyzw"]
                object_loc = annotation["location"]
                object_quat = annotation["quaternion_xyzw"]
        #dataset camera intrinsics
        camera_intrinsics = np.array(annotations[0]['camera_intrinsics'])
        #depth image used to extract table plane
        depth_name = YCB_dataset_dir +'data/'+file_name+'/'+"{:06d}".format(IND)+"-depth.png"
        #label image used for background extraction
        label_name = YCB_dataset_dir +'data/'+file_name+'/'+"{:06d}".format(IND)+"-label.png"
        
        # For now, we get our z from ground truth pose 
        gt_object_to_cam = R.from_quat(object_quat)
        gt_object_to_cam = gt_object_to_cam.as_matrix()
        gt_obj_2_cam = np.zeros((4,4))
        gt_obj_2_cam[:3,:3] = gt_object_to_cam
        gt_obj_2_cam[:,3] = object_loc+[1]

        #needs to offset by 1/2 height of the object to render the object on table plane.(without the offset, center of the object will be rendered at table plane while we need the bottom of the objet on table)
        #should be using model of the object (.ply file) to get offset, but somehow does not work very well, maybe need to transform model. Currently we assume perfect knowledge of the offset and used groud truth z pose
        table_2_cam = table_pose.get_real_transform(depth_name,label_name,camera_intrinsics,gt_obj_2_cam)
        #run c++ files to render images
        generate_data.generate(YCB_dataset_dir,file_name,IND, table_2_cam,search_object)
        #run cost function on the rendered images
        data_folder = data_dir+str(IND)+"/"
        pose_file = data_dir+str(IND)+"/data/pose.txt"
        #output files(error and accuracy) written with in ycb.run
        output_pose = ycb.run(output_f,error_f,search_object,data_folder,pose_file,model_dir,IND,gt_pose)

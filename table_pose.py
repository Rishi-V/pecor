import numpy as np
import pcl
import pcl.pcl_visualization
from scipy.spatial.transform import Rotation as R

def get_world_point(point,camera_intrinsics) :
        camera_fx_reciprocal_ = 1.0 /camera_intrinsics[0, 0]
        camera_fy_reciprocal_ = 1.0 /camera_intrinsics[1, 1]

        world_point = np.zeros(3)

        world_point[2] = point[2]
        world_point[0] = (point[0] -camera_intrinsics[0,2]) * point[2] * (camera_fx_reciprocal_)
        world_point[1] = (point[1] -camera_intrinsics[1,2]) * point[2] * (camera_fy_reciprocal_)

        return world_point
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def get_table_pose(depth_img_path, label_image_path,camera_intrinsics):
        '''
            Creates a point cloud in camera frame and calculates table pose using RANSAC
        '''
        import cv2
        from PIL import Image
        depth_image = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
        depth_image1 = cv2.imread(depth_img_path,0)
        label = cv2.imread(label_image_path,0)
        K_inv = np.linalg.inv(camera_intrinsics)
        points_3d = []
        count = 0
        cloud = pcl.PointCloud()
        depth_image_pil = np.asarray(Image.open(depth_img_path), dtype=np.float16)
        for x in range(depth_image.shape[1]):
            for y in range(depth_image.shape[0]):
                # depends on the depth image, may need to adjust the threshold
                if label[y,x] == 0 and depth_image[y,x] > 30 and depth_image1[y,x] < 40:
                    point = np.array([x,y,depth_image[y,x]/10000])
                    w_point = get_world_point(point,camera_intrinsics)
                    # Table cant be above camera
                    if w_point[1] < 0.0:
                        continue
                    points_3d.append(w_point.tolist() )
                    count += 1

        points_3d = np.array(points_3d).astype(np.float32)
        cloud.from_array(points_3d)
        seg = cloud.make_segmenter()
        # Optional
        seg.set_optimize_coefficients (True)
        # Mandatory
        seg.set_model_type (pcl.SACMODEL_PLANE)
        seg.set_method_type (pcl.SAC_RANSAC)
        seg.set_distance_threshold (0.03)
        inliers, model = seg.segment()
        cloud1 = pcl.PointCloud()
        cloud1.from_array(points_3d[inliers])
        # for visualization of the extracted table plane
        # visualcolor1 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud, 0, 255, 0)
        # visualcolor2=pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud1,255,0,0)
        # vs=pcl.pcl_visualization.PCLVisualizering
        # vss1=pcl.pcl_visualization.PCLVisualizering()#Initialize an object, here is a very important step
        # vs.AddPointCloud_ColorHandler(vss1,cloud,visualcolor1,id=b'cloud1',viewport=0)
        # vs.AddPointCloud_ColorHandler(vss1, cloud1, visualcolor2, id=b'cloud', viewport=0)
        
        # v = True
        # while not vs.WasStopped(vss1):
        #     vs.Spin(vss1)
        rotation_matrix = rotation_matrix_from_vectors([0,0,1], [model[0], model[1], model[2]])
        angles = []
        yaw = np.arctan(model[1]/model[0])
        pitch = np.arctan(model[2]/model[1])+np.pi/2
        roll = 0
        quat = R.from_euler('xyz',[roll,pitch,yaw]).as_quat()
        inlier_points = points_3d[inliers]
        location = np.mean(inlier_points[:,:3], axis=0)
        print("Table location : {}".format(location))
        table_to_cam = R.from_quat([quat[1], quat[0], quat[2], quat[3]])
        table_to_cam = table_to_cam.as_matrix()
        table_2_cam = np.zeros((4,4))
        # table_2_cam[:3,:3] = table_to_cam
        table_2_cam[:3,:3] = rotation_matrix
        table_2_cam[:,3] = [location[0], location[1], location[2], 1]
        cam_to_table = np.linalg.inv(table_2_cam)
    

        return table_2_cam
        
def get_real_transform(depth_name,label_name,camera_intrinsics,gt_obj_2_cam):
        #table pose respect to camera frame
        table_2_cam = get_table_pose(depth_name,label_name,camera_intrinsics)
        #camera pose respect to table frame
        cam_to_table = np.linalg.inv(table_2_cam)
        #object to table frame
        object_to_table = np.matmul(cam_to_table, gt_obj_2_cam)
        #since we consider the center, object needs to move in Z direction 1/2 of its height
        move = np.matrix([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, object_to_table[2,3]],
                    [0, 0, 0, 1 ]])
        real_transform = np.matmul(table_2_cam,move)
        return real_transform
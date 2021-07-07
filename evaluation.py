from scipy.spatial.transform import Rotation as R
import numpy as np
# from plyfile import PlyData, PlyElement
import scipy
from sklearn.metrics import pairwise_distances_chunked, pairwise_distances_argmin_min


def calculate_ADD(gt, perch_res,model_dir,search_object,f= None,IND= 0, ycb = False, dope = False):
    # need to transfer to a matrix 
    r_gt = R.from_quat([gt[3], gt[4], gt[5], gt[6]])
    r_gt = r_gt.as_matrix()
    if not ycb:
        camera_matrix = np.array([[0, 0, 1],
                                [-1, 0,0],
                                [0, -1,0]])
        r_gt = np.matmul(r_gt, camera_matrix)
    trans_gt = np.zeros((4,4))
    trans_gt[:3,:3] = r_gt
    if ycb:
        trans_gt[:,3] = [gt[0],gt[1],gt[2],1]
    else:
        trans_gt[:,3] = [gt[0]/100,gt[1]/100,gt[2]/100,1]

    r_p = R.from_quat([perch_res[3], perch_res[4], perch_res[5], perch_res[6]])
    r_p = r_p.as_matrix()
    trans_p = np.zeros((4,4))
    trans_p[:3,:3] = r_p
    trans_p[:,3] = [perch_res[0]/100,perch_res[1]/100,perch_res[2]/100,1]
    if dope:
        trans_p[:,3] = [perch_res[0]/100,perch_res[1]/100,gt[2],1]

    #ADD and ADD-S are based on the point cloud of the obejct
    cloud = np.loadtxt(model_dir+search_object+'/points.xyz')
    cloud = np.hstack((cloud, np.ones((cloud.shape[0], 1))))
    # print(cloud)
    transformed_cloud_gt = np.matmul(trans_gt, np.transpose(cloud))
    transformed_cloud_p = np.matmul(trans_p, np.transpose(cloud))
    # print(trans_gt)
    # print(trans_p)
    # Mean of corresponding points
    mean_dist = np.linalg.norm(transformed_cloud_gt-transformed_cloud_p, axis=0)
    # print(mean_dist.shape)
    mean_dist_add = np.sum(mean_dist)/cloud.shape[0]
    print("Average pose distance - ADD (in m) : {}".format(mean_dist_add))

    # Do ADD-S for symmetric objects or every object if true
    transformed_cloud_gt = np.transpose(transformed_cloud_gt)
    transformed_cloud_p = np.transpose(transformed_cloud_p)
    # For below func matrix should be samples x features
    pairwise_distances = pairwise_distances_argmin_min(
        transformed_cloud_gt, transformed_cloud_p, metric='euclidean', metric_kwargs={'n_jobs':6}
    )
    # Mean of nearest points
    mean_dist_add_s = np.mean(pairwise_distances[1])
    print("Average pose distance - ADD-S (in m) : {}".format(mean_dist_add_s))
    if f:
        f.write(str(IND)+":"+ "{}".format(mean_dist_add)+","+ "{}".format(mean_dist_add_s)+"\n")

#include <cuda_renderer/renderer.h>
#include <hist_prioritize/hist.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <angles/angles.h>
#include <ros/ros.h>
#include "std_msgs/String.h"
#include <bits/stdc++.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <queue>
// #include <filesystem>
// #include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "./file_paths.h"
#ifdef CUDA_ON
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#endif
#define SQR(x) ((x)*(x))
#define POW2(x) SQR(x)
#define POW3(x) ((x)*(x)*(x))
#define POW4(x) (POW2(x)*POW2(x))
#define POW7(x) (POW3(x)*POW3(x)*(x))
#define DegToRad(x) ((x)*M_PI/180)
#define RadToDeg(x) ((x)/M_PI*180)

class Pose
{
    public:
        Pose(double x, double y, double z, double roll, double pitch, double yaw);
        Eigen::Isometry3d GetTransform() const;
    
        double x_ = 0.0;
        double y_ = 0.0;
        double z_ = 0.0;
        double roll_ = 0.0;
        double pitch_ = 0.0;
        double yaw_ = 0.0;
};

Eigen::Isometry3d Pose::GetTransform() const {
  const Eigen::AngleAxisd roll_angle(roll_, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitch_angle(pitch_, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yaw_angle(yaw_, Eigen::Vector3d::UnitZ());
  Eigen::Quaterniond quaternion;
  quaternion = yaw_angle * pitch_angle * roll_angle;
  quaternion.normalize();
  const Eigen::Isometry3d transform(Eigen::Translation3d(x_, y_, z_) * quaternion);
  return transform;
}

Pose::Pose(double x, double y, double z, double roll, double pitch,
                   double yaw) : x_(x), y_(y), z_(z),
  roll_(angles::normalize_angle_positive(roll)),
  pitch_(angles::normalize_angle_positive(pitch)),
  yaw_(angles::normalize_angle_positive(yaw)) {
};



class color_only
{
public:

  color_only(std::string search_object,std::string img,const std::string r00,const std::string r01,const std::string r02,const std::string r03, const std::string r10,const std::string r11,const std::string r12,const std::string r13, const std::string r20,const std::string r21,const std::string r22,const std::string r23);
  
  std::vector<cuda_renderer::Model> models;
  std::vector<float> mode_trans;
  cv::Mat cam_intrinsic;
  Eigen::Matrix4d cam_intrinsic_eigen;
  Eigen::Matrix4d table_to_cam;
  int width;
  int height;
  cv::Mat background_image;
  cv::Mat origin_image;
  cv::Mat cv_input_color_image;

  float x_min,x_max,y_min,y_max;
  float res,theta_res;
  float prune_percent;
  int render_size;
  
  cuda_renderer::Model::mat4x4 proj_mat;
  std::vector<cuda_renderer::Model::mat4x4> trans_mat;
  std::vector<Pose> Pose_list;
  std::vector<float> gpu_bb;
  std::vector<std::vector<float> > gpu_cam_m;

  std::string vis_folder;
  //functions 
  void setinput();
  // main render function
  void generate_image(const std::string img, const std::string pic_idx);
  // only used to do visualization, can render images with specific pose
  void generate_gt(const std::string img, const std::string pic_idx,const std::string r00,const std::string r01,const std::string r02,const std::string r03, const std::string r10,const std::string r11,const std::string r12,const std::string r13, const std::string r20,const std::string r21,const std::string r22,const std::string r23);

  
  ~color_only();
  
};

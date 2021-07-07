#include <color_only/color_render.h>
color_only::~color_only()
{

}
color_only::color_only(std::string search_object, std::string label,const std::string r00,const std::string r01,const std::string r02,const std::string r03, const std::string r10,const std::string r11,const std::string r12,const std::string r13, const std::string r20,const std::string r21,const std::string r22,const std::string r23){
    //setup
    std::string prefix = MODEL_FOLDER + search_object +"/";
    width = 640;
    height = 480;
    float kCameraFX=1.066778e+03;
    float kCameraFY=1.067487e+03;
    float kCameraCX=3.129869e+02;
    float kCameraCY=2.413109e+02;

    cam_intrinsic=(cv::Mat_<float>(3,3) << kCameraFX, 0.0, kCameraCX, 0.0, kCameraFY, kCameraCY, 0.0, 0.0, 1.0);
    cam_intrinsic_eigen << kCameraFX, 0.0, kCameraCX,0.0, 0.0, kCameraFY, kCameraCY,0.0, 0.0, 0.0, 1.0,0.0,0.0,0.0,0.0,0.0;
    proj_mat = cuda_renderer::compute_proj(cam_intrinsic, width, height);
    // table_to_cam from python script
    table_to_cam.matrix() <<  std::stod(r00),std::stod(r01),std::stod(r02),std::stod(r03),
    std::stod(r10),std::stod(r11),std::stod(r12),std::stod(r13),
    std::stod(r20),std::stod(r21),std::stod(r22),std::stod(r23),
    0,0,0,1;                

    
    //search space and resolution
    x_min = -0.3;
    x_max = 0.3;
    y_min = -0.3;
    y_max = 0.3;

    res = 0.01;
    theta_res =0.5;
    prune_percent = 0.4;
    // only used when visualize certain pose
    vis_folder = "/home/jessy/perch-clean/";


    cuda_renderer::Model model(prefix+"textured.ply");
    models.push_back(model);
    background_image = cv::imread(label, cv::IMREAD_COLOR);   // background subtraction YCB use label(degmentation image)
    // 8 bounding box points 
    gpu_bb.push_back(models[0].bbox_min.x);gpu_bb.push_back(models[0].bbox_min.y);gpu_bb.push_back(models[0].bbox_min.z);
    gpu_bb.push_back(models[0].bbox_max.x);gpu_bb.push_back(models[0].bbox_min.y);gpu_bb.push_back(models[0].bbox_min.z);
    gpu_bb.push_back(models[0].bbox_min.x);gpu_bb.push_back(models[0].bbox_max.y);gpu_bb.push_back(models[0].bbox_min.z);
    gpu_bb.push_back(models[0].bbox_max.x);gpu_bb.push_back(models[0].bbox_max.y);gpu_bb.push_back(models[0].bbox_min.z);
    gpu_bb.push_back(models[0].bbox_min.x);gpu_bb.push_back(models[0].bbox_min.y);gpu_bb.push_back(models[0].bbox_max.z);
    gpu_bb.push_back(models[0].bbox_max.x);gpu_bb.push_back(models[0].bbox_min.y);gpu_bb.push_back(models[0].bbox_max.z);
    gpu_bb.push_back(models[0].bbox_min.x);gpu_bb.push_back(models[0].bbox_max.y);gpu_bb.push_back(models[0].bbox_max.z);
    gpu_bb.push_back(models[0].bbox_max.x);gpu_bb.push_back(models[0].bbox_max.y);gpu_bb.push_back(models[0].bbox_max.z);
    // camera matirx to vector for GPU
    std::vector<float> cam_r1;
    std::vector<float> cam_r2;
    std::vector<float> cam_r3;
    Eigen::Matrix4d gpu_cam = cam_intrinsic_eigen*table_to_cam;
    cam_r1.push_back(gpu_cam(0,0));cam_r1.push_back(gpu_cam(0,1));cam_r1.push_back(gpu_cam(0,2));cam_r1.push_back(gpu_cam(0,3));
    cam_r2.push_back(gpu_cam(1,0));cam_r2.push_back(gpu_cam(1,1));cam_r2.push_back(gpu_cam(1,2));cam_r2.push_back(gpu_cam(1,3));
    cam_r3.push_back(gpu_cam(2,0));cam_r3.push_back(gpu_cam(2,1));cam_r3.push_back(gpu_cam(2,2));cam_r3.push_back(gpu_cam(2,3));
    
    gpu_cam_m.push_back(cam_r1);
    gpu_cam_m.push_back(cam_r2);
    gpu_cam_m.push_back(cam_r3);
    std::cout<<"SetInput Finished!!!"<<std::endl;
}


void color_only::generate_image(const std::string msg, const std::string pic_idx)
{
    // for creating folders
    std::string folder = IMAGE_FOLDER_PATH+"/"+pic_idx+"/";
    const char * folder_name = folder.c_str();
    mkdir(folder_name,0777); 
    folder = IMAGE_FOLDER_PATH+"/"+pic_idx+"/data/";
    const char * folder_name1 = folder.c_str();
    mkdir(folder_name1,0777); 
    std::string folder_bg = IMAGE_BG_FOLDER_PATH+"/"+pic_idx+"/";
    const char * folder_bg_name = folder_bg.c_str();
    mkdir(folder_bg_name,0777); 
    cv::Mat image = cv::imread(msg, cv::IMREAD_COLOR);;

    origin_image.release();
    cv_input_color_image.release();
    trans_mat.clear();
    Pose_list.clear();
    origin_image = image;
    std::cout<<"Image Loaded"<<std::endl;

    //ycb background substraction
    background_image = (background_image > 0);
    cv::Mat mask;
    mask = background_image;
    cv::cvtColor(mask,mask,CV_BGR2GRAY);
    cv::threshold(mask,mask,15,255,CV_THRESH_BINARY);
    cv::bitwise_or(origin_image, origin_image,cv_input_color_image, mask=mask);
    
    cv::Mat gray_input;
    cv::cvtColor(cv_input_color_image,gray_input,CV_BGR2GRAY);
    //find 2D bounding box for colored region
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( gray_input, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );
    std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
    imwrite(folder+"-1.jpg", cv_input_color_image);   // RV: TODO save input image
    std::vector<int> boundRect;
    int bound_count = 0;
    
    for( size_t i = 0; i < contours.size(); i++ )
    {
        // might need to denoise 
        cv::approxPolyDP( contours[i], contours_poly[i], 3, true );
        cv::Rect cur_rect = boundingRect( contours_poly[i] );
        if(cur_rect.width>10 && cur_rect.height>10){
            // add margin when detecting
            boundRect.push_back(cur_rect.x-40);
            boundRect.push_back(cur_rect.y-40);
            boundRect.push_back(cur_rect.width+80);
            boundRect.push_back(cur_rect.height+80);
            bound_count++;
            std::cout<<cur_rect.x<<","<<cur_rect.y<<","<<cur_rect.width<<","<<cur_rect.height<<std::endl;
        }        
    }
    for(auto b:boundRect){
        std::cout<<b<<","<<std::endl;
    }
    
    std::cout<<"number of color regions"<<bound_count<<std::endl;
    boundRect.insert(boundRect.begin(),bound_count);
    // process to vectors for GPU
    std::vector<uint8_t> r_v;
    std::vector<uint8_t> g_v;
    std::vector<uint8_t> b_v;

    int non_zero =0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            cv::Vec3b elem_rgb = cv_input_color_image.at<cv::Vec3b>(y, x);
            r_v.push_back(elem_rgb[2]);
            g_v.push_back(elem_rgb[1]);
            b_v.push_back(elem_rgb[0]);
            if(elem_rgb[0]!=0 || elem_rgb[1]!=0 || elem_rgb[2]!=0){
              non_zero +=1;
            }
            
        }
    }
    std::vector<std::vector<uint8_t>> observed_rgb;
    observed_rgb.push_back(r_v);
    observed_rgb.push_back(g_v);
    observed_rgb.push_back(b_v);
    std::cout<<"total pixel number"<< non_zero<<std::endl;
    // only used for determine whether a pose rendered on colored region 
    //prune_percent is adjustable, note that here we use 8 point from bounding box to determine percent rendered on backgound, no rendering at this stage
    hist_prioritize::s_pose his_result =  hist_prioritize::valid_poses(width,height,
                                                                    x_min,x_max,y_min,y_max,
                                                                    0.0,2 * M_PI,
                                                                    res,theta_res,non_zero,prune_percent,
                                                                    observed_rgb,gpu_cam_m,gpu_bb,0,boundRect);
    
    for(int i =0; i < his_result.ps.size(); i ++){
        //ycb
        // pose respect to table center
        Pose cur = Pose(his_result.ps[i].x, his_result.ps[i].y, 0, 0, 0, his_result.ps[i].theta);
        Pose_list.push_back(cur);
    }
    // can be adjusted relative to GPU size
    int render_size = 500;
    int total_render_num = Pose_list.size();
    int num_render = (total_render_num-1)/render_size+1;
    int count_p = 0;
    std::vector<cuda_renderer::Model::mat4x4> cur_transform;
    std::ofstream pose_file;
    pose_file.open (folder+"pose.txt",std::ios_base::app);
    for(int i =0; i <num_render; i ++){
        auto last = std::min(total_render_num, i*render_size + render_size);
        std::vector<Pose>::const_iterator start = Pose_list.begin() + i*render_size;
        std::vector<Pose>::const_iterator finish = Pose_list.begin() + last;
        std::vector<Pose> cur_Pose_list(start,finish);
        
        for(int n = 0; n <cur_Pose_list.size();n++){
            //change the pose to mat4x4 defined by the renderer
            Pose cur = cur_Pose_list[n];
            Eigen::Matrix4d transform;
            transform = cur.Pose::GetTransform().matrix().cast<double>();
            Eigen::Matrix4d pose_in_cam = table_to_cam*transform;
            cuda_renderer::Model::mat4x4 mat4;
            mat4.a0 = pose_in_cam(0,0)*100;
            mat4.a1 = pose_in_cam(0,1)*100;
            mat4.a2 = pose_in_cam(0,2)*100;
            mat4.a3 = pose_in_cam(0,3)*100;
            mat4.b0 = pose_in_cam(1,0)*100;
            mat4.b1 = pose_in_cam(1,1)*100;
            mat4.b2 = pose_in_cam(1,2)*100;
            mat4.b3 = pose_in_cam(1,3)*100;
            mat4.c0 = pose_in_cam(2,0)*100;
            mat4.c1 = pose_in_cam(2,1)*100;
            mat4.c2 = pose_in_cam(2,2)*100;
            mat4.c3 = pose_in_cam(2,3)*100;
            mat4.d0 = pose_in_cam(3,0);
            mat4.d1 = pose_in_cam(3,1);
            mat4.d2 = pose_in_cam(3,2);
            mat4.d3 = pose_in_cam(3,3);
            cur_transform.push_back(mat4);
        }
        //render all images results in vector of vector 
        std::vector<std::vector<uint8_t>> result_gpu = cuda_renderer::render_cuda(models[0].tris,cur_transform,
                                                                                    width, height,proj_mat);

        cv::Mat cur_mat = cv::Mat(height,width,CV_8UC3);
        cv::Mat cur_mat_with_background = cv::Mat(height,width,CV_8UC3);
        //change back to cv mat for visulization and saving the image to file
        for(int n = 0; n <cur_Pose_list.size(); n ++){
            for(int i = 0; i < height; i ++){
                for(int j = 0; j <width; j ++){
                    int index = n*width*height+(i*width+j);
                    int red = result_gpu[0][index];
                    int green = result_gpu[1][index];
                    int blue = result_gpu[2][index];
                    if(blue == 0 && green == 0 && red == 0){
                        cur_mat.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0,0);
                        cur_mat_with_background.at<cv::Vec3b>(i, j) = cv_input_color_image.at<cv::Vec3b>(i, j);
                    }else{
                        cur_mat.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green,red);
                        cur_mat_with_background.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green,red);
                    }
                    
                }
            }
            //write image to folder
            std::ostringstream out;
            out << std::internal << std::setfill('0') << std::setw(8) << count_p;
            imwrite(folder+out.str()+".jpg", cur_mat); 
            imwrite(folder_bg_name+out.str()+".jpg", cur_mat_with_background);
            // write rendered pose to file
            Eigen::Matrix3f angle_mat;
            angle_mat.matrix()<<cur_transform[n].a0/100,cur_transform[n].a1/100,cur_transform[n].a2/100,
                                cur_transform[n].b0/100,cur_transform[n].b1/100,cur_transform[n].b2/100,
                                cur_transform[n].c0/100,cur_transform[n].c1/100,cur_transform[n].c2/100;
            Eigen::Quaternionf q(angle_mat);
            pose_file<< count_p <<","<<cur_transform[n].a3<<","<<cur_transform[n].b3<<","<<cur_transform[n].c3<<","<<q.x()<<","<<q.y()<<","<<q.z()<<","<<q.w()<<"\n";
            count_p = count_p+1;
            cur_transform.clear();
        }
  }

}

// used to visualize/save images in certain pose, mainly used for debugging. same with the main function but only render 1 pose
void color_only::generate_gt(const std::string msg, const std::string pic_idx,const std::string r00,const std::string r01,const std::string r02,const std::string r03, const std::string r10,const std::string r11,const std::string r12,const std::string r13, const std::string r20,const std::string r21,const std::string r22,const std::string r23)
{
    std::string folder = IMAGE_FOLDER_PATH+"/"+pic_idx+"/";
    const char * folder_name = folder.c_str();
    mkdir(folder_name,0777); 
    folder = IMAGE_FOLDER_PATH+"/"+pic_idx+"/data/";
    const char * folder_name1 = folder.c_str();
    mkdir(folder_name1,0777); 
    std::string folder_bg = IMAGE_BG_FOLDER_PATH+"/"+pic_idx+"/";
    const char * folder_bg_name = folder_bg.c_str();
    mkdir(folder_bg_name,0777); 
    cv::Mat image = cv::imread(msg, cv::IMREAD_COLOR);;
    auto start_l = std::chrono::steady_clock::now();
    origin_image.release();
    cv_input_color_image.release();
    trans_mat.clear();
    origin_image = image;
    std::cout<<"Image Loaded"<<std::endl;
 
    //ycb baskground subtraction
    background_image = (background_image > 0);
    cv::Mat mask;
    mask = background_image;
    cv::cvtColor(mask,mask,CV_BGR2GRAY);
    cv::threshold(mask,mask,15,255,CV_THRESH_BINARY);
    cv::bitwise_or(origin_image, origin_image,cv_input_color_image, mask=mask);
    cv_input_color_image = image;
    cv::Mat gray_input;
    cv::cvtColor(cv_input_color_image,gray_input,CV_BGR2GRAY);
    

    std::vector<cuda_renderer::Model::mat4x4> cur_transform;
    
    //render specific pose in cam
    Eigen::Matrix4d pose_in_cam;
    pose_in_cam.matrix()<< std::stod(r00),std::stod(r01),std::stod(r02),std::stod(r03),
    std::stod(r10),std::stod(r11),std::stod(r12),std::stod(r13),
    std::stod(r20),std::stod(r21),std::stod(r22),std::stod(r23),
    0,0,0,1;  
                                             

    cuda_renderer::Model::mat4x4 mat4;
    mat4.a0 = pose_in_cam(0,0)*100;
    mat4.a1 = pose_in_cam(0,1)*100;
    mat4.a2 = pose_in_cam(0,2)*100;
    mat4.a3 = pose_in_cam(0,3)*100;
    mat4.b0 = pose_in_cam(1,0)*100;
    mat4.b1 = pose_in_cam(1,1)*100;
    mat4.b2 = pose_in_cam(1,2)*100;
    mat4.b3 = pose_in_cam(1,3)*100;
    mat4.c0 = pose_in_cam(2,0)*100;
    mat4.c1 = pose_in_cam(2,1)*100;
    mat4.c2 = pose_in_cam(2,2)*100;
    mat4.c3 = pose_in_cam(2,3)*100;
    mat4.d0 = pose_in_cam(3,0);
    mat4.d1 = pose_in_cam(3,1);
    mat4.d2 = pose_in_cam(3,2);
    mat4.d3 = pose_in_cam(3,3);
    cur_transform.push_back(mat4);
  
    std::vector<std::vector<uint8_t>> result_gpu = cuda_renderer::render_cuda(models[0].tris,cur_transform,
                                                                                width, height,proj_mat);
    cv::Mat cur_mat = cv::Mat(height,width,CV_8UC3);
    cv::Mat cur_mat_with_background = cv::Mat(height,width,CV_8UC3);
    cv::Mat cur_mask = cv::Mat(height,width,CV_8UC1);

    for(int i = 0; i < height; i ++){
        for(int j = 0; j <width; j ++){
            int index = 0*width*height+(i*width+j);
            int red = result_gpu[0][index];
            int green = result_gpu[1][index];
            int blue = result_gpu[2][index];
            if(blue == 0 && green == 0 && red == 0){
                cur_mat_with_background.at<cv::Vec3b>(i, j) = cv_input_color_image.at<cv::Vec3b>(i, j);
                cur_mat.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0,0);
                cur_mask.at<unsigned char>(i, j) = 0;
            }else{
                cur_mat.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green,red);
                cur_mat_with_background.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green,red);
                cur_mask.at<unsigned char>(i, j) = 255;
            }
            
        }
    }
    
    std::ostringstream out;
    imwrite(vis_folder + pic_idx+".jpg", cv_input_color_image);  
    imwrite(vis_folder + pic_idx+"_render.jpg", cur_mat); 
    imwrite(vis_folder + pic_idx+"_bg.jpg", cur_mat_with_background);
    imwrite(vis_folder + pic_idx+"_mask.jpg", cur_mask);
}


int main(int argc, char **argv)
{
    
    std::string address = argv[1];
    std::string pic_idx = argv[2];
    std::string r00 = argv[3];
    std::string r01 = argv[4];
    std::string r02 = argv[5];
    std::string r03 = argv[6];
    std::string r10 = argv[7];
    std::string r11 = argv[8];
    std::string r12 = argv[9];
    std::string r13 = argv[10];
    std::string r20 = argv[11];
    std::string r21 = argv[12];
    std::string r22 = argv[13];
    std::string r23 = argv[14];
    std::string label = argv[15];
    std::string search_object = argv[16];
    std::cout<< address;
    color_only test(search_object,label,r00,r01,r02,r03, r10,r11,r12,r13, r20,r21,r22,r23);
    test.generate_image(address,pic_idx);
    // test.generate_gt(address,pic_idx,r00,r01,r02,r03, r10,r11,r12,r13, r20,r21,r22,r23);
    
    return 0;
}


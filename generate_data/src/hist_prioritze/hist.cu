#include "./include/hist_prioritize/hist.h"
// #include <math.h> 
#include "cuda_fp16.h"

// #include <numeric> 
#define SQR(x) ((x)*(x))
#define POW2(x) SQR(x)
#define POW3(x) ((x)*(x)*(x))
#define POW4(x) (POW2(x)*POW2(x))
#define POW7(x) (POW3(x)*POW3(x)*(x))
#define DegToRad(x) ((x)*M_PI/180)
#define RadToDeg(x) ((x)/M_PI*180)

namespace hist_prioritize {

template<typename T>
device_vector_holder<T>::~device_vector_holder(){
    __free();
}

template<typename T>
void device_vector_holder<T>::__free(){
    if(valid){
        cudaFree(__gpu_memory);
        valid = false;
        __size = 0;
    }
}

template<typename T>
device_vector_holder<T>::device_vector_holder(size_t size_, T init)
{
    __malloc(size_);
    thrust::fill(begin_thr(), end_thr(), init);
}

template<typename T>
void device_vector_holder<T>::__malloc(size_t size_){
    if(valid) __free();
    cudaMalloc((void**)&__gpu_memory, size_ * sizeof(T));
    __size = size_;
    valid = true;
}

template<typename T>
device_vector_holder<T>::device_vector_holder(size_t size_){
    __malloc(size_);
}

template class device_vector_holder<int>;

void print_cuda_memory_usage(){
    // show memory usage of GPU

    size_t free_byte ;
    size_t total_byte ;
    auto cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

    if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

struct max2zero_functor{

    max2zero_functor(){}

    __host__ __device__
    int32_t operator()(const int32_t& x) const
    {
      return (x==INT_MAX)? 0: x;
    }
};

struct pose_functor{

    pose_functor(){}

    __host__ __device__
    pose operator()(const pose& x) const
    {
      return x;
    }
};

// used to determin if the rendered pose is in colored region of the input image
__device__ bool inside_ROI(float x_min,float x_max,float y_min,float y_max, const int* roi){
    int count = roi[0];
    for(int i =0; i < count; i ++){
        int x = roi[i*4+1];
        int y = roi[i*4+2];
        int width = roi[i*4+3];
        int height = roi[i*4+4];
        if(x_min>= x && y_min >= y  && x_max <= x+width &&y_max <= y+height)
            return true;
    }
    return false;
}



__global__ void find_poses(int32_t* out_score,int32_t* valid_p,pose* output_p,
                                   const int width, const int height,
                                   const float x_min, const float y_min, const float angle_min,const float x_max, const float y_max, const float angle_max,
                                   const float trans_res,const float angle_res,const int x_num,const int y_num,const int angle_num, const float alpha,
                                   const uint8_t* r_ob, const uint8_t* g_ob,const uint8_t* b_ob,
                                   const float* cam_r1,const float* cam_r2,const float* cam_r3,
                                   const float* bb, const int* roi, const float z)
{
    size_t angle_i = blockIdx.y;
    size_t trans_i = blockIdx.x*blockDim.x + threadIdx.x;
    int32_t output_index = (angle_num)*trans_i+angle_i;
    float x = x_min+(int)floorf(trans_i/y_num)*trans_res;
    float y = y_min+(trans_i%y_num)*trans_res;
    float theta = angle_min+angle_i*angle_res;
    // for every pose, calculate 2D pixel position from 3D point and camera matrix
    if(output_index<x_num*y_num*angle_num){
        float min_x=10000;
        float max_x=-10000;
        float min_y=10000;
        float max_y=-10000;
        // printf("%d,%d: %f,%f,%f;\n",output_index,x_num*y_num*angle_num,x,y,theta);
        for(int i =0; i <8;i++){
            float cur_x = bb[i*3];
            float cur_y = bb[i*3+1];
            float cur_z = bb[i*3+2];

            float res_x = cur_x*(cam_r1[0]*cos(theta)+cam_r1[1]*(sin(theta)))+
                          cur_y*(cam_r1[0]*(-sin(theta))+cam_r1[1]*(cos(theta)))+
                          cur_z*cam_r1[2]+
                          cam_r1[0]*x+cam_r1[1]*y+cam_r1[2]*z+cam_r1[3];
            float res_y = cur_x*(cam_r2[0]*cos(theta)+cam_r2[1]*(sin(theta)))+
                          cur_y*(cam_r2[0]*(-sin(theta))+cam_r2[1]*(cos(theta)))+
                          cur_z*cam_r2[2]+
                          cam_r2[0]*x+cam_r2[1]*y+cam_r2[2]*z+cam_r2[3];
            float res_z = cur_x*(cam_r3[0]*cos(theta)+cam_r3[1]*(sin(theta)))+
                          cur_y*(cam_r3[0]*(-sin(theta))+cam_r3[1]*(cos(theta)))+
                          cur_z*cam_r3[2]+
                          cam_r3[0]*x+cam_r3[1]*y+cam_r3[2]*z+cam_r3[3];
            float bx = res_x/res_z;
            float by = res_y/res_z;
            if(bx<min_x) min_x = bx;
            if(bx>max_x) max_x = bx;
            if(by<min_y) min_y = by;
            if(by>max_y) max_y = by;
        }
        int bg_pixel = 0;
        if(min_x>=0 && min_x<width&&max_x>=0 && max_x<width&&
            // here +50 is used to include poses that part of it is outside of image, same margin can be added for width
            min_y>=0 && min_y<height&&max_y>=0 && max_y<height+50 && inside_ROI(min_x,max_x,min_y,max_y,roi)){
            //inside the bounding box, count how many colored pixles are in input image
            for(int cur_y=min_y;cur_y<=max_y;cur_y++){
                for(int cur_x = min_x;cur_x<=max_x;cur_x++){
                    int cur_ind = cur_y*width+cur_x;
                    uint8_t r_value = r_ob[cur_ind];
                    uint8_t g_value = g_ob[cur_ind];
                    uint8_t b_value = b_ob[cur_ind];
                    if(r_value==0 && g_value==0 && b_value == 0){
                        bg_pixel+=1;
                    }
                }
            }
            //prune away poses that render too much on background
            if(bg_pixel > (max_x-min_x)*(max_y-min_y)*alpha){
                out_score[output_index] = 1;
            }else{
 
                int32_t& valid_add = valid_p[0];
                atomicAdd(&valid_add,1);
                out_score[output_index] = 0;
                output_p[output_index].x = x;
                output_p[output_index].y = y;
                output_p[output_index].theta = theta;
            }
            

        }else{
            out_score[output_index] =1;

        }
        
    }
    
}

s_pose valid_poses(const int width, const int height,const float x_min,const float x_max,
                              const float y_min,const float y_max,
                              const float theta_min,const float theta_max,
                              const float trans_res, const float angle_res,
                              const int32_t ob_pixel_num,
                              const float prune_percent,
                              const std::vector<std::vector<uint8_t>>& observed,
                              const std::vector<std::vector<float> >& cam_matrix,
                              const std::vector<float>& bounding_boxes,
                              const float z,
                              const std::vector<int>& color_region
                              
                              )
{

    float elapsed1=0;
    float elapsed2=0;
    cudaEvent_t start1, stop1,start2,stop2;

    HANDLE_ERROR(cudaEventCreate(&start1));
    HANDLE_ERROR(cudaEventCreate(&stop1));

    HANDLE_ERROR( cudaEventRecord(start1, 0));

    const size_t threadsPerBlock = 256;
    
    float x_range = x_max-x_min;
    float y_range = y_max-y_min;
    float angle_range = theta_max-theta_min;
    int x_num =(int)floor(x_range / trans_res* 10+0.5)/10+1;
    int y_num =(int)floor(y_range / trans_res*10+0.5)/10+1;
    int angle_num = (int)floor(angle_range/angle_res*10+0.5)/10+1;

    thrust::device_vector<pose> d_output_p(x_num*y_num*angle_num);
    thrust::device_vector<int> output_score(x_num*y_num*angle_num, 0);
    thrust::device_vector<uint8_t> d_r_ob = observed[0];
    thrust::device_vector<uint8_t> d_g_ob = observed[1];
    thrust::device_vector<uint8_t> d_b_ob = observed[2];
    thrust::device_vector<float> cam_row1 = cam_matrix[0];
    thrust::device_vector<float> cam_row2 = cam_matrix[1];
    thrust::device_vector<float> cam_row3 = cam_matrix[2];
    thrust::device_vector<int> c_region = color_region;
    thrust::device_vector<int> d_valid(1, 0);

    thrust::device_vector<float> bb = bounding_boxes;

    {

        int32_t* output_s = thrust::raw_pointer_cast(output_score.data());
        pose* output_p = thrust::raw_pointer_cast(d_output_p.data());
        int32_t* d_valid_p = thrust::raw_pointer_cast(d_valid.data());
        uint8_t* r_ob = thrust::raw_pointer_cast(d_r_ob.data());
        uint8_t* g_ob = thrust::raw_pointer_cast(d_g_ob.data());
        uint8_t* b_ob = thrust::raw_pointer_cast(d_b_ob.data());
        float* cam_r1 = thrust::raw_pointer_cast(cam_row1.data());
        float* cam_r2 = thrust::raw_pointer_cast(cam_row2.data());
        float* cam_r3 = thrust::raw_pointer_cast(cam_row3.data());
        float* bounding_box = thrust::raw_pointer_cast(bb.data());
        int* roi = thrust::raw_pointer_cast(c_region.data());
        // glm::mat4* a = thrust::raw_pointer_cast(cam_matrix.data());


        dim3 numBlocks((x_num*y_num + threadsPerBlock - 1) / threadsPerBlock, angle_num);
        find_poses<<<numBlocks, threadsPerBlock>>>(output_s,d_valid_p,output_p,
                                                        width,height,
                                                        x_min, y_min, theta_min,x_max, y_max, theta_max,
                                                        trans_res,angle_res,x_num,y_num,angle_num,prune_percent,
                                                        r_ob,g_ob,b_ob,
                                                        cam_r1,cam_r2,cam_r3,bounding_box,roi,z);
        cudaDeviceSynchronize();
    }
    

    std::vector<int> x(x_num*y_num*angle_num);
    std::vector<int> y(x_num*y_num*angle_num);
    std::vector<int> theta(x_num*y_num*angle_num);
    
    std::vector<int> valid_num(1);
    {

        thrust::transform(d_valid.begin(), d_valid.end(),
                          d_valid.begin(), max2zero_functor());
        thrust::copy(d_valid.begin(), d_valid.end(), valid_num.begin());
        
        thrust::sort_by_key(output_score.begin(), output_score.end(), d_output_p.begin());
        thrust::transform(d_output_p.begin(), d_output_p.end(),
                          d_output_p.begin(),pose_functor());
        
    }

    std::vector<pose> v(valid_num[0]);
    thrust::copy(d_output_p.begin(), d_output_p.begin()+valid_num[0], v.begin());
    std::vector<std::vector<int> > res;
    s_pose poses;
    poses.ps = v;
    return poses;
}


}

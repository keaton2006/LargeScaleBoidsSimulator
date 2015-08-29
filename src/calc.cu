/*
 * LargeBoidsSimulator
 *
 * Yhoichi Mototake
 */


#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <vector_types.h>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#include "calc.h"
#include "param.h"

/////////////
/*Variables*/
/////////////////////////////////////

float3 *speed = NULL;
float3 *tmp_speed = NULL;
float4 *tmp_position = NULL;

float3 *original_speed = NULL;
float4 *original_position = NULL;

float4 *tmp_position_for_fill = NULL;
float4 *dptr = NULL;

int* coord = NULL;
int* coord_particle = NULL;

DataStruct* data_for_calc;

Paramstruct* param_host;
Paramstruct* param_device;

cudaStream_t stream0;
cudaStream_t stream1;
/////////////////////////////////////

//////////////////////////////////////////
/*Definitions of inline device functions*/
/////////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float4 operator+(const float4 &a, const float4 &b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y,a.z + b.z);
}
inline __host__ __device__ double3 operator+(const double3 &a, const float4 &b)
{
    return make_double3(a.x + (double)b.x, a.y + (double)b.y,a.z + (double)b.z);
}
inline __host__ __device__ double3 operator+(const double3 &a, const double4 &b)
{
    return make_double3(a.x + b.x, a.y + b.y,a.z + b.z);
}
inline __host__ __device__ double3 operator+(const double3 &a, const double3 &b)
{
    return make_double3(a.x + b.x, a.y + b.y,a.z + b.z);
}
inline __host__ __device__ double4 operator-(const float4 &a, const float4 &b)
{
    return make_double4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ double4 operator+(const double4 &a, const double4 &b)
{
    return make_double4(a.x + b.x, a.y + b.y,a.z + b.z,a.w + b.w);
}
inline __host__ __device__ float4 operator-(const float4 &a, const float &b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ float4 operator+(const float4 &a, const float &b)
{
    return make_float4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ double4 operator/(const double4 &a, const double &b)
{
    return make_double4(a.x / b, a.y / b, a.z / b,  a.w / b);
}
inline __host__ __device__ float3 operator/(const float3 &a, const float &b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ double3 operator/(const double3 &a, const int &b)
{
    return make_double3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ float4 operator*(const float &a, const float4 &b)
{
    return make_float4(a * b.x, a * b.y, a * b.z,a * b.w);
}
inline __host__ __device__ float3 operator*(const float &a, const float3 &b)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}
inline __host__ __device__ double4 operator*(const double &a, const double4 &b)
{
    return make_double4(a * b.x, a * b.y, a * b.z,a * b.w);
}
inline __host__ __device__ float dot(float4 a, float4 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}
inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
inline __host__ __device__ float3 cast_float3(double4 b)
{
	float3 a;
    a.x = (float)b.x;
    a.y = (float)b.y;
    a.z = (float)b.z;
	return a;
}
inline __host__ __device__ float3 cast_float3(double3 b)
{
	float3 a;
    a.x = (float)b.x;
    a.y = (float)b.y;
    a.z = (float)b.z;
	return a;
}
inline __host__ __device__ float3 cast_float3(float4 b)
{
	float3 a;
    a.x = (float)b.x;
    a.y = (float)b.y;
    a.z = (float)b.z;
	return a;
}
inline __host__ __device__ float4 cast_float4(float3 b)
{
	float4 a;
    a.x = (float)b.x;
    a.y = (float)b.y;
    a.z = (float)b.z;
    a.w = 0.0;
	return a;
}
/////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////
/*Definitions of global device functions*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//main calculation function
__global__ void calc_core(Paramstruct* param,float4 *original_position ,float3 *original_speed , float4 *tmp_position,float4 *tmp_position_for_fill, unsigned int width, unsigned int height,float3 *tmp_speed, int *coord,int* coord_particle)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int index = y*width+x;


	float4 position;
    position = original_position[index];


    float3 velocity;
    velocity = original_speed[index];

    __shared__ Paramstruct get_param;
    get_param = *param;

    int neighboursCount;
    neighboursCount = 0;
    int neighboursCount_ave;
    neighboursCount_ave = 0;


    double3 neighboursAvgSpeed;
    neighboursAvgSpeed = make_double3(0, 0, 0);


    double4 neighboursAvgPosition;
    neighboursAvgPosition = make_double4(0, 0, 0, 0);

    double3 separationForce;
    separationForce = make_double3(0, 0, 0);

    float4 p;
    double d;


    int Nd = 2*get_param.field_size*(int)(1/get_param.mesh_size);
    int max_mesh_index = (int)(get_param.max_distance/get_param.mesh_size+0.999);

    for(int count_x=-max_mesh_index;count_x<=max_mesh_index;count_x++){
       	for(int count_y=-max_mesh_index;count_y<=max_mesh_index;count_y++){
       		for(int count_z=-max_mesh_index;count_z<=max_mesh_index;count_z++){

    			int z_index = (int)((1/get_param.mesh_size)*(position.z+get_param.field_size)) + count_z;
   				int y_index = (int)((1/get_param.mesh_size)*(position.y+get_param.field_size)) + count_y;

				int x_index = (int)((1/get_param.mesh_size)*(position.x+get_param.field_size)) + count_x;


				int x_flag = 0;
				int y_flag = 0;
				int z_flag = 0;
				if (z_index < 0){
					z_index = Nd + z_index;
					z_flag = -1;
				}
				if(z_index >= Nd){
					z_index = z_index - Nd;
					z_flag = 1;
				}
				if(y_index < 0){
					y_index = Nd + y_index;
					y_flag = -1;
				}
				if(y_index >= Nd){
					y_index = y_index - Nd;
					y_flag = 1;
				}
				if(x_index < 0){
					x_index = Nd + x_index;
					x_flag = -1;
				}
				if(x_index >= Nd){
					x_index = x_index - Nd;
					x_flag = 1;
				}


    						int coord_index = (z_index*Nd*Nd + y_index*Nd + x_index);
    						int tmp_index;
    						tmp_index = coord[coord_index];
    						int count = 0;

    						while(count!=-1){
    							//if(tmp_index == index) printf("1+");
   								if(tmp_index!=-1){
									p = original_position[tmp_index];
#if PERIODIC == 1
									if (z_flag == 1){
										p.z = p.z + 2*FIELD_SIZE;
									}
									if(z_flag == -1){
										p.z = p.z - 2*FIELD_SIZE;
									}
									if(y_flag == 1){
										p.y = p.y + 2*FIELD_SIZE;
									}
									if(y_flag == -1){
										p.y = p.y - 2*FIELD_SIZE;
									}
									if(x_flag == 1){
										p.x = p.x + 2*FIELD_SIZE;
									}
									if(x_flag == -1){
										p.x = p.x - 2*FIELD_SIZE;
									}
#endif

							        float k2 = ((p.x-position.x)*velocity.x + (p.y-position.y)*velocity.y + (p.z-position.z)*velocity.z)/sqrt((p.x-position.x)*(p.x-position.x)+(p.y-position.y)*(p.y-position.y)+(p.z-position.z)*(p.z-position.z))/sqrt(velocity.x*velocity.x+velocity.y*velocity.y+velocity.z*velocity.z);
							        if(k2 > get_param.max_angle){ //when tmp_index = index: k2 = nan that's why no problem

										float dx = abs(position.x - p.x);
										if (dx < get_param.max_distance){
											float dy = abs(position.y - p.y);
											if (dy < get_param.max_distance){
												float dz = abs(position.z - p.z);
												if (dz < get_param.max_distance){

													d = sqrt(dx*dx + dy*dy + dz*dz);
													if (d < get_param.max_distance){

														//alignment
														if(d < get_param.sight_distance_alignment && k2 > get_param.sight_angle_alignment){
															neighboursCount_ave++;
															neighboursAvgSpeed = neighboursAvgSpeed + make_double3(original_speed[tmp_index].x,original_speed[tmp_index].y,original_speed[tmp_index].z);
														}

														//cohesion
														if (d < get_param.sight_distance_cohesion && k2 > get_param.sight_angle_cohesion){
															if(d>0){
																neighboursCount++;
																neighboursAvgPosition = neighboursAvgPosition + make_double4(p.x,p.y,p.z,p.w);
															}else{
																printf("position of two particles are completely matched, there are probability of bug existing.\n");
															}
														}

														//separation
														if (d < get_param.min_distance && k2 > get_param.sight_angle_separation) {
															if(d>0){
																separationForce = separationForce + ((position - p))/d;
															}else{
																printf("position of two particles are completely matched, there are probability of bug existing.\n");
															}
														}


													}
												}
											}
										}
										count += 1;
							        }
									tmp_index = coord_particle[tmp_index];
									if(tmp_index==-1){
										count = -1;
									}

    							}else{
    									count = -1;
    							}

   							}



    		}
    	}

    }

    float tmp_coeff = EPS*get_param.min_distance*get_param.w_min_distance;
    velocity = velocity + tmp_coeff*(cast_float3(separationForce));

    if(neighboursCount_ave > 0){
    	neighboursAvgSpeed = neighboursAvgSpeed / neighboursCount_ave;

    	velocity = velocity + EPS*get_param.w_neighbour_speed*(cast_float3(neighboursAvgSpeed) - velocity);
    }
    if(neighboursCount > 0){

    	neighboursAvgPosition = neighboursAvgPosition / neighboursCount;

    	float3 coh_v;
    	coh_v = cast_float3(neighboursAvgPosition) - cast_float3(position);


        //float norm = 1;//sqrt(coh_x*coh_x+coh_y*coh_y+coh_z*coh_z);


        velocity = velocity + EPS*get_param.w_neighbour_distance*coh_v;///norm;

    }



#if NOISE == 1
    curandState s;
    curand_init(10, index, 0, &s);
    float3 noise_rand = make_float3(2*(curand_uniform(&s)-0.5),2*(curand_uniform(&s)-0.5),2*(curand_uniform(&s)-0.5));
    velocity = velocity + EPS*get_param.w_noise* get_param.max_speed * noise_rand;
#endif


    float check_speed = sqrt(velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z);
    if(check_speed > get_param.max_speed){
    	float norm = 1/(get_param.max_speed)*check_speed;
    	velocity = velocity / norm;

    }else if(check_speed < get_param.min_speed){
    	float norm = 1/(get_param.min_speed)*check_speed;
    	velocity = velocity / norm;
    }

    position = position + EPS * cast_float4(velocity);


    float scale = (float)get_param.field_size;
#if PERIODIC == 1
    if (position.x >= scale) {
        position.x = -2*scale + position.x;
    }else if (position.x < -scale) {
        position.x = 2*scale + position.x;
    }
    if (position.y >= scale) {
        position.y = -2*scale + position.y;
    }else if (position.y < -scale) {
        position.y = 2*scale + position.y;
    }
    if (position.z >= scale) {
            position.z = -2*scale + position.z;
    }else if (position.z < -scale) {
            position.z = 2*scale + position.z;
    }
#else
    if (position.x >= scale) {
        	velocity.x = -velocity.x;
            position.x = 2*scale - position.x;
        }else if (position.x < -scale) {
          	velocity.x = -velocity.x;
            position.x = -2*scale - position.x;
        }
    if (position.y >= scale) {
        	velocity.y = -velocity.y;
            position.y = 2*scale - position.y;
        }else if (position.y < -scale) {
          	velocity.y = -velocity.y;
            position.y = -2*scale - position.y;
        }
    if (position.z >= scale) {
    	velocity.z = -velocity.z;
        position.z = 2*scale - position.z;
    }else if (position.z < -scale) {
      	velocity.z = -velocity.z;
        position.z = -2*scale - position.z;
    }
#endif

    tmp_position_for_fill[index] = original_position[index];
    tmp_position[index] = position;
    tmp_speed[index] = velocity;

}

	//reload original variables
__global__ void reload(float4 *original_position ,float3 *original_speed , float4 *tmp_position, unsigned int width, unsigned int height,float3 *tmp_speed)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int index = y*width+x;

	original_position[index] = tmp_position[index];
	original_speed[index] = tmp_speed[index];

}

	//sent data for
__global__ void sent_data(float4 *pos,float4 *tmp_position, unsigned int width, unsigned int height,float3 *tmp_speed,float3 *speed)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int index = y*width+x;

	pos[index] = tmp_position[index];
	speed[index] = tmp_speed[index];
}

	//set initial value of position and speed
__global__ void prepare_calc(Paramstruct* param,float4 *original_position, unsigned int width, unsigned int height, float3 *original_speed)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int index = y*width+x;

	// calculate uv coordinates
	float u = x / (float) width;
	float v = y / (float) height;
	u = u*INIT_BOIDS_SIZE - INIT_BOIDS_SIZE/2.0f;
	v = v*INIT_BOIDS_SIZE - INIT_BOIDS_SIZE/2.0f;
	__syncthreads();
	curandState s;
	curand_init(width, index, 0, &s);
	original_position[index] = make_float4(INIT_BOIDS_SIZE*(2*(curand_uniform(&s)-0.5)), INIT_BOIDS_SIZE*(2*(curand_uniform(&s)-0.5)), INIT_BOIDS_SIZE*(2*(curand_uniform(&s)-0.5)),1.0f);//1*random(index), 1.0f);

	original_speed[index] = make_float3(0.05*2*(curand_uniform(&s)-0.5),0.05*2*(curand_uniform(&s)-0.5),0.05*2*(curand_uniform(&s)-0.5));
}

	//fill -1 in lattice array
__global__ void fill_space(float4 *tmp_position_for_fill, unsigned int width, unsigned int height, int *coord,int* coord_particle)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int index = y*width+x;
	int Nd = 2*FIELD_SIZE*(int)(1/MESH_SIZE);
	int coord_index_pre = ((int)((1/MESH_SIZE)*(tmp_position_for_fill[index].z+FIELD_SIZE)))*Nd*Nd + ((int)((1/MESH_SIZE)*(tmp_position_for_fill[index].y+FIELD_SIZE)))*Nd + ((int)((1/MESH_SIZE)*(tmp_position_for_fill[index].x+FIELD_SIZE)));
	coord[coord_index_pre] = -1;
}

	//regist particle id for lattice array
__global__ void divide_space(float4 *original_position, unsigned int width, unsigned int height, int *coord,int* coord_particle)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int index = y*width+x;
	int Nd = 2*FIELD_SIZE*(int)(1/MESH_SIZE);
	int coord_index = ((int)((1/MESH_SIZE)*(original_position[index].z+FIELD_SIZE)))*Nd*Nd + ((int)((1/MESH_SIZE)*(original_position[index].y+FIELD_SIZE)))*Nd + ((int)((1/MESH_SIZE)*(original_position[index].x+FIELD_SIZE)));

	if(atomicCAS(&coord[coord_index],-1,index)==-1){
		coord_particle[index] = -1;
	}else{
		int ok_flag = 0;
		while(ok_flag==0){
			int tmp_index = coord[coord_index];
			if(atomicCAS(&coord[coord_index],tmp_index,index)==tmp_index){
				coord_particle[index] = tmp_index;
				ok_flag = 1;
			}else{
				//printf(" ERROR ");
			}
		}
	}

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void prepare(float4 *pos, unsigned int mesh_width,
                    unsigned int mesh_height, float3 *speed)
{

	// execute the kernel
	int block_y;
	if(mesh_height >= 32){
		block_y = 32;
	}else{
		block_y = 1;
	}
    dim3 block(32, block_y, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);

	prepare_calc<<< grid, block,0,stream0>>>(param_device,tmp_position, mesh_width, mesh_height, tmp_speed);
	prepare_calc<<< grid, block,0,stream0>>>(param_device,original_position, mesh_width, mesh_height, original_speed);
	//zero fill
	int Nd = 2*FIELD_SIZE*(int)(1/MESH_SIZE)+1;
	cudaStreamSynchronize(stream0);
	thrust::device_ptr<int> coord_thrust(coord);
	thrust::fill(coord_thrust,coord_thrust+Nd*Nd*Nd,-1);
	divide_space<<< grid, block,0,stream0>>>(original_position, mesh_width, mesh_height, coord,coord_particle);
}

void setparam(){

	param_host = (Paramstruct*)malloc(sizeof(Paramstruct));
	if(SIGHT_DISTANCE_COHESION >= SIGHT_DISTANCE_ALIGNMENT){
		param_host->max_distance = SIGHT_DISTANCE_COHESION;//distance_cohesion
	}else{
		param_host->max_distance = SIGHT_DISTANCE_ALIGNMENT;
	}
	printf("max_distance=%f\n",param_host->max_distance);
	param_host->sight_distance_cohesion = SIGHT_DISTANCE_COHESION;
	param_host->sight_distance_alignment = SIGHT_DISTANCE_ALIGNMENT;
	param_host->min_distance = MIN_DISTANCE;

	if(SIGHT_ANGLE_COHESION < SIGHT_ANGLE_ALIGNMENT && SIGHT_ANGLE_COHESION < SIGHT_ANGLE_SEPARATION){
		param_host->max_angle = SIGHT_ANGLE_COHESION;
	}else if(SIGHT_ANGLE_ALIGNMENT < SIGHT_ANGLE_SEPARATION){
		param_host->max_angle = SIGHT_ANGLE_ALIGNMENT;
	}else{
		param_host->max_angle = SIGHT_ANGLE_SEPARATION;
	}
	printf("max_dangle=%f\n",param_host->max_angle);
	param_host->sight_angle_separation = SIGHT_ANGLE_SEPARATION;
	param_host->sight_angle_alignment = SIGHT_ANGLE_ALIGNMENT;
	param_host->sight_angle_cohesion = SIGHT_ANGLE_COHESION;

	param_host->w_neighbour_speed = W_NEIGHBOUR_SPEED;
	param_host->w_neighbour_distance = W_NEIGHBOUR_DISTANCE;
	param_host->w_min_distance = W_MIN_DISTANCE;
	param_host->w_noise = W_NOISE;
	param_host->min_speed = MIN_SPEED;
	param_host->max_speed = MAX_SPEED;

	param_host->field_size = FIELD_SIZE;
	param_host->mesh_size = MESH_SIZE;


    checkCudaErrors(cudaMemcpy(param_device, param_host, sizeof(Paramstruct), cudaMemcpyHostToDevice));
}


void launch_calc(float4 *pos, unsigned int mesh_width,
                   unsigned int mesh_height, float3 *speed)
{

	int block_y;
	if(mesh_height >= 32){
		block_y = 32;
	}else{
		block_y = 1;
	}
    dim3 block(32, block_y, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);

	cudaStreamSynchronize(stream0);
	calc_core<<< grid, block,0,stream0>>>(param_device, original_position ,original_speed ,tmp_position,tmp_position_for_fill, mesh_width, mesh_height ,tmp_speed, coord,coord_particle);
	cudaStreamSynchronize(stream0);
	reload<<< grid, block,0,stream0>>>(original_position ,original_speed , tmp_position, mesh_width, mesh_height,tmp_speed);
#if DEBUG == 1
	//zero fill
	thrust::device_ptr<int> coord_thrust(coord);
	thrust::fill(coord_thrust,coord_thrust+Nd*Nd,-1);
#endif
	fill_space<<< grid, block,0,stream0>>>(tmp_position_for_fill, mesh_width, mesh_height, coord,coord_particle);
	cudaStreamSynchronize(stream0);
	divide_space<<< grid, block,0,stream0>>>(original_position, mesh_width, mesh_height, coord,coord_particle);
	cudaStreamSynchronize(stream0);
	cudaHostGetDevicePointer( &pos, data_for_calc->a, 0 );
	cudaHostGetDevicePointer( &speed, data_for_calc->b, 0 );
	sent_data<<< grid, block,0,stream1>>>(pos,original_position, mesh_width, mesh_height ,tmp_speed,speed);
}



bool malloc_val(int argc, char **argv, char *ref_file)
{
	int Nd = 2*FIELD_SIZE*(int)(1/MESH_SIZE)+1;

	checkCudaErrors(cudaMalloc((void **)&param_device, sizeof(Paramstruct)));

	checkCudaErrors(cudaMalloc((void **)&original_speed, mesh_width*mesh_height*sizeof(float3)));
	checkCudaErrors(cudaMalloc((void **)&original_position, mesh_width*mesh_height*sizeof(float4)));

	checkCudaErrors(cudaMalloc((void **)&tmp_speed, mesh_width*mesh_height*sizeof(float3)));
	checkCudaErrors(cudaMalloc((void **)&tmp_position, mesh_width*mesh_height*sizeof(float4)));

	checkCudaErrors(cudaMalloc((void **)&tmp_position_for_fill, mesh_width*mesh_height*sizeof(float4)));
	checkCudaErrors(cudaMalloc((void **)&coord_particle, mesh_width*mesh_height*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&coord, Nd*Nd*Nd*sizeof(int)));

	cudaError_t result;
	result = cudaStreamCreate(&stream1);
	result = cudaStreamCreate(&stream0);
	printf("error=%d\n",result);

	return true;
}

void run()
{
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr = NULL;
	launch_calc(dptr, mesh_width, mesh_height, speed);
}

void preparefunc()
{
	// map OpenGL buffer object for writing from CUDA
	prepare(dptr, mesh_width, mesh_height, speed);
}

void* routine( void *pvoidData ){
	printf("calc start\n");
	data_for_calc = (DataStruct*)pvoidData;
    char *ref_file = NULL;

    int *argc = data_for_calc->pArgc;
    char **argv = data_for_calc->pArgv;

	malloc_val(argc[0], argv, ref_file);
	setparam();
	preparefunc();

	FILE* save_fp;

	int n = 0; //dummy variable to erase the "unreachable" warning
    while(n != EOF){

    	run();

    	if(data_for_calc->time%1000<=100){
    		if(data_for_calc->time%1000==0){
    			char file_name[100] = "param0_";

    			sprintf(file_name,"param%d_%ld.csv",PARAM_NUM,data_for_calc->time);

    			save_fp = fopen(file_name,"w");
    		}

    		fprintf(save_fp,"%ld",data_for_calc->time);
    			for(int count=0;count<N;count++){
    				fprintf(save_fp,",%f,%f,%f",data_for_calc->a[count].x,data_for_calc->a[count].y,data_for_calc->a[count].z);
    				fprintf(save_fp,",%f,%f,%f",data_for_calc->b[count].x,data_for_calc->b[count].y,data_for_calc->b[count].z);
    			}
    			fprintf(save_fp,"\n");
    			if(data_for_calc->time%1000==100){
    				fclose(save_fp);
    			}
    	}

    	data_for_calc->time += 1;
    }
    cudaStreamDestroy( stream0 );
    cudaStreamDestroy( stream1 );

    printf("finish_process\n");

    return 0;
}

/*
 * LargeBoidsSimulator
 *
 * Yhoichi Mototake
 */

#ifndef PARAM_H_
#define PARAM_H_

#define DEBUG 0
#define FRONT 1
#define DRAW_CUDA 1  //0:no  1:draw by cuda  2:drawing controlled by cpu

#define PARAM_NUM 1

const unsigned int mesh_width    = 32; //32 no baisuu wo suisyou
const unsigned int mesh_height   = 4;
#define N mesh_width*mesh_height

#define WORLD_PERSPECTIVE 10
#define FIELD_SIZE 0.5  //minimum 0.5
#define INIT_BOIDS_SIZE 0.1

#define SIGHT_DISTANCE_COHESION 0.1//0.02
#define SIGHT_DISTANCE_ALIGNMENT 0.1//0.02
#define MIN_DISTANCE 0.01

#define SIGHT_ANGLE_SEPARATION -0.8
#define SIGHT_ANGLE_ALIGNMENT -0.8
#define SIGHT_ANGLE_COHESION -0.8

#define W_NEIGHBOUR_SPEED 0.012
#define W_NEIGHBOUR_DISTANCE 0.05//0.01//0.0015//0.15
#define W_MIN_DISTANCE 0.007//*MINDISTANCE
#define NOISE 0
#define W_NOISE 0.0
#define MIN_SPEED 0.0045
#define MAX_SPEED 0.0047

#define EPS 1

#define MESH_SIZE 0.01

#define PERIODIC 1

#endif /* PARAM_H_ */

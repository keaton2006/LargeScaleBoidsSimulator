/*
 * LargeBoidsSimulator
 *
 * Yhoichi Mototake
 */

#ifndef CALC_H_
#define CALC_H_

#include "gpus_controller.h"
#include <stdio.h>
#include "time.h"




void* routine( void *pvoidData );

typedef struct{
	float max_distance;//distance_cohesion
	float sight_distance_cohesion;
	float sight_distance_alignment;
	float min_distance;

	float max_angle;
	float sight_angle_separation;
	float sight_angle_alignment;
	float sight_angle_cohesion;

	float w_neighbour_speed;
	float w_neighbour_distance;
	float w_min_distance;
	float w_noise;
	float min_speed;
	float max_speed;

	float field_size;
	float mesh_size;
} Paramstruct;

#endif /* CALC_H_ */

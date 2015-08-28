/*
 * LargeBoidsSimulator
 *
 * Yhoichi Mototake
 */

#ifndef GPUS_CONTROLLER_H_
#define GPUS_CONTROLLER_H_

#include "calc.h"
#include "draw.h"
#include "param.h"

// CUDA helper functions
#include <vector_types.h>




typedef struct{
    int     deviceID;
    int     size;
    int     offset;
    float4*   a;
    float3*   b;
    float   returnValue;
    int		*pArgc;
    char 	**pArgv;
    long int time;

}DataStruct;


int calc_main( int argc, char **argv );

#endif /* GPUS_CONTROLLER_H_ */

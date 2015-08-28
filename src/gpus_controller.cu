/*
 * LargeBoidsSimulator
 *
 * Yhoichi Mototake
 */
//#include "./book.h"
#include "gpus_controller.h"
#include <stdio.h>
#include <stdlib.h>
#include "./book.h"

#if DRAW_CUDA == 2
#include <pthread.h>
#endif
int calc_main( int argc, char **argv ) {


#if DRAW_CUDA == 1
    int deviceCount;
    HANDLE_ERROR( cudaGetDeviceCount( &deviceCount ) );
    if (deviceCount < 2) {
        printf( "We need at least two compute 1.0 or greater "
                "devices, but only found %d\n", deviceCount );
        return 0;
    }

    cudaDeviceProp  prop;
    for (int i=0; i<2; i++) {
        HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
        if (prop.canMapHostMemory != 1) {
            printf( "Device %d can not map memory.\n", i );
            return 0;
        }
    }
#else
#endif

    float4 *a;
    float3 *b;

    HANDLE_ERROR( cudaSetDeviceFlags( cudaDeviceMapHost ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&a, N*sizeof(float4),
                              cudaHostAllocWriteCombined |
                              cudaHostAllocPortable |
                              cudaHostAllocMapped ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&b, N*sizeof(float3),
                              cudaHostAllocWriteCombined |
                              cudaHostAllocPortable      |
                              cudaHostAllocMapped ) );

    // fill in the host memory with data
    for (int i=0; i<N; i++) {
        a[i].x = i;
        b[i].y = i;
    }

    // preparation for multithread
    DataStruct  data[2];
    data[0].deviceID = 0;
    data[0].offset = 0;
    data[0].size = N/2;
    data[0].a = a;
    data[0].b = b;

    data[1].deviceID = 1;
    data[1].offset = N/2;
    data[1].size = N/2;
    data[1].a = a;
    data[1].b = b;
    data[1].pArgc = &argc;
    data[1].pArgv = argv;
    data[1].time = 0;
#if DRAW_CUDA == 1
    HANDLE_ERROR( cudaSetDevice( 1 ) );
    CUTThread   thread = start_thread( rundraw, &(data[1]) );
#else
    pthread_t th_draw;
    pthread_create(&th_draw,NULL,draw,&(data[1]));
#endif

#if DRAW_CUDA == 1
    HANDLE_ERROR( cudaSetDevice( 0 ) );
#endif
    routine( &(data[1]) );

    printf("101\n");
#if DRAW_CUDA == 1
    end_thread( thread );
#endif
    // free memory on the CPU side
    HANDLE_ERROR( cudaFreeHost( a ) );
    HANDLE_ERROR( cudaFreeHost( b ) );

    cudaDeviceReset();
    printf("finish all\n");
    return 0;
}

/*
 * LargeBoidsSimulator
 *
 * Yhoichi Mototake
 */

#include "png.h"

#include "draw.h"
#include "param.h"

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

#if DRAW_CUDA == 1
// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#endif

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms


DataStruct  *data;
////////////////////////////////////////////////////////////////////////////////

// constants
const unsigned int window_width  = 512*2;
const unsigned int window_height = 512*2;


#if DRAW_CUDA == 1

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;
#endif
float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

//for analy
int Na = 1000;
float Pg_tmp[1000];
float Mg_tmp[1000];

float perspective = 0;
float point_size = 0;


FILE *fp;


////////////////////////////////////////////////////////////////////////////////
#if DRAW_CUDA == 1

// declaration, forward
bool runTest(int argc, char **argv, char *ref_file);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);
#endif
// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);
#if DRAW_CUDA == 1

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);
void runAutoTest(int devID, char **argv, char *ref_file);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

const char *sSDKsample = "simpleGL (VBO)";
#endif
////////////////////////
// save capture image //
////////////////////////



void capture()
{
    const char filepath[] = "./output.png";
    png_bytep raw1D;
    png_bytepp raw2D;
    int i;
    int width = glutGet(GLUT_WINDOW_WIDTH);
    int height = glutGet(GLUT_WINDOW_HEIGHT);

    // 構造体確保
    FILE *fp = fopen(filepath, "wb");
    png_structp pp = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop ip = png_create_info_struct(pp);
    // 書き込み準備
    png_init_io(pp, fp);
    png_set_IHDR(pp, ip, width, height,
        8, // 8bit以外にするなら変える
        PNG_COLOR_TYPE_RGBA, // RGBA以外にするなら変える
        PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    // ピクセル領域確保
    raw1D = (png_bytep)malloc(height * png_get_rowbytes(pp, ip));
    raw2D = (png_bytepp)malloc(height * sizeof(png_bytep));
    for (i = 0; i < height; i++)
        raw2D[i] = &raw1D[i*png_get_rowbytes(pp, ip)];
    // 画像のキャプチャ
    glReadBuffer(GL_FRONT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // 初期値は4
    glReadPixels(0, 0, width, height,
            GL_RGBA, // RGBA以外にするなら変える
            GL_UNSIGNED_BYTE, // 8bit以外にするなら変える
            (void*)raw1D);
    // 上下反転
    for (i = 0; i < height/ 2; i++){
        png_bytep swp = raw2D[i];
        raw2D[i] = raw2D[height - i - 1];
        raw2D[height - i - 1] = swp;
    }
    // 書き込み
    png_write_info(pp, ip);
    png_write_image(pp, raw2D);
    png_write_end(pp, ip);
    // 開放
    png_destroy_write_struct(&pp, &ip);
    fclose(fp);
    free(raw1D);
    free(raw2D);

    printf("write out screen capture to '%s'\n", filepath);
}


#if DRAW_CUDA == 1

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_vbo_kernel(float4* a,float4 *pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    // write output vertex
    pos[y*width+x] = a[y*width+x];//make_float4(u, w, 0.01*y, 1.0f);
    //printf("dev_a=%f\n",a[0]);
}


void launch_kernel(float4 *pos, unsigned int mesh_width,
                   unsigned int mesh_height, float time)
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
    //int     size = data->size;
    float4   *a;
    float3 *b;
    float4   *dev_a;
    float *dev_b;

    // allocate memory on the CPU side
    a = data->a;
    b = data->b;
    //printf("dev_a=%f\n",a[0]);
    // allocate the memory on the GPU
    cudaHostGetDevicePointer( &dev_a, a, 0 );
    cudaHostGetDevicePointer( &dev_b, b, 0 );

    // offset 'a' and 'b' to where this GPU is gets it data
    //dev_a += data->offset;
    //dev_b += data->offset;

    simple_vbo_kernel<<< grid, block>>>(dev_a,pos, mesh_width, mesh_height, time);
}

bool checkHW(char *name, const char *gpuType, int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    strcpy(name, deviceProp.name);

    if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

int findGraphicsGPU(char *name)
{
    int nGraphicsGPU = 0;
    int deviceCount = 0;
    bool bFoundGraphics = false;
    char firstGraphicsName[256], temp[256];

    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("> FAILED %s sample finished, exiting...\n", sSDKsample);
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("> There are no device(s) supporting CUDA\n");
        return false;
    }
    else
    {
        printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        bool bGraphics = !checkHW(temp, (const char *)"Tesla", dev);
        printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);

        if (bGraphics)
        {
            if (!bFoundGraphics)
            {
                strcpy(firstGraphicsName, temp);
            }

            nGraphicsGPU++;
        }
    }

    if (nGraphicsGPU)
    {
        strcpy(name, firstGraphicsName);
    }
    else
    {
        strcpy(name, "this hardware");
    }

    return nGraphicsGPU;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
void* rundraw(void* pvoidData)
{
    data = (DataStruct*)pvoidData;
    char *ref_file = NULL;

    pArgc = data->pArgc;
    pArgv = data->pArgv;




#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    printf("%s starting...\n", sSDKsample);

    if (pArgc[0] > 1)
    {
        if (checkCmdLineFlag(pArgc[0], (const char **)pArgv, "file"))
        {
            // In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
            getCmdLineArgumentString(pArgc[0], (const char **)pArgv, "file", (char **)&ref_file);
        }
    }

    printf("\n");

    runTest(pArgc[0], pArgv, ref_file);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    //cudaDeviceReset();
    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    if((fp=fopen("./test", "wb"))==NULL) {
      printf("Cannot open file.\n");
      exit(1);
    }
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    glewInit();

    if (! glewIsSupported("GL_VERSION_2_0 "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, WORLD_PERSPECTIVE);

    SDK_CHECK_ERROR_GL();

    return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, char *ref_file)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    // command line mode only
    if (ref_file != NULL)
    {
        // This will pick the best possible CUDA capable device
        int devID = findCudaDevice(argc, (const char **)argv);

        // create VBO
        checkCudaErrors(cudaMalloc((void **)&d_vbo_buffer, mesh_width*mesh_height*4*sizeof(float)));

        // run the cuda part
        runAutoTest(devID, argv, ref_file);

        // check result of Cuda step
        checkResultCuda(argc, argv, vbo);

        cudaFree(d_vbo_buffer);
        d_vbo_buffer = NULL;
    }
    else
    {
        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        if (false == initGL(&argc, argv))
        {
            return false;
        }

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        if (checkCmdLineFlag(argc, (const char **)argv, "device"))
        {
            if (gpuGLDeviceInit(argc, (const char **)argv) == -1)
            {
                return false;
            }
        }
        else
        {
            cudaGLSetGLDevice(1);
        }

        // register callbacks
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif

        // create VBO
        createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

        // run the cuda part
        runCuda(&cuda_vbo_resource);

        // start rendering mainloop
        glutMainLoop();
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                         *vbo_resource));

    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

    // execute the kernel
    //    dim3 block(8, 8, 1);
    //    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    //    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

    launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif

void sdkDumpBin2(void *data, unsigned int bytes, const char *filename)
{
    printf("sdkDumpBin: <%s>\n", filename);
    FILE *fp;
    FOPEN(fp, filename, "wb");
    fwrite(data, bytes, 1, fp);
    fflush(fp);
    fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runAutoTest(int devID, char **argv, char *ref_file)
{
    char *reference_file = NULL;
    void *imageData = malloc(mesh_width*mesh_height*sizeof(float));

    // execute the kernel
    launch_kernel((float4 *)d_vbo_buffer, mesh_width, mesh_height, g_fAnim);

    cudaDeviceSynchronize();
    getLastCudaError("launch_kernel failed");

    checkCudaErrors(cudaMemcpy(imageData, d_vbo_buffer, mesh_width*mesh_height*sizeof(float), cudaMemcpyDeviceToHost));

    sdkDumpBin2(imageData, mesh_width*mesh_height*sizeof(float), "simpleGL.bin");
    reference_file = sdkFindFilePath(ref_file, argv[0]);

    if (reference_file &&
        !sdkCompareBin2BinFloat("simpleGL.bin", reference_file,
                                mesh_width*mesh_height*sizeof(float),
                                MAX_EPSILON_ERROR, THRESHOLD, pArgv[0]))
    {
        g_TotalErrors++;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

    // unregister this buffer object with CUDA
    cudaGraphicsUnregisterResource(vbo_res);

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}
#endif
////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    /*if(data->time == 1){
    	capture();
    }*/
    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, WORLD_PERSPECTIVE+perspective);

#if DRAW_CUDA == 1

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource);
#endif
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor((float)(156.0/255.0),(float)(167.0/255.0),(float)(186.0/255.0),0);
    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
#if DRAW_CUDA == 1

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
#endif

    glVertexPointer(4, GL_FLOAT, 0, 0);

    ////////////////////////
    //calc order parameter//
    ////////////////////////
    /*int Np = mesh_width*mesh_height;
    float Pg = 0;
    float Pg_x = 0;
    float Pg_y = 0;
    float Pg_z = 0;
    float Mg = 0;
    float Mg_x = 0;
    float Mg_y = 0;
    float Mg_z = 0;
    float x = 0;
    float y = 0;
    float z = 0;
    fprintf(fp, "%f,%f,%f,", data->a[1000].x, data->a[1000].y,data->a[1000].z);
    for(int count=0;count < Np;count++){
    	Pg_x += data->b[count].x;
    	Pg_y += data->b[count].y;
    	Pg_z += data->b[count].z;

    	x += data->a[count].x/Np;
    	y += data->a[count].y/Np;
    	z += data->a[count].z/Np;
    }
    for(int count=0;count < Np;count++){
    	Mg_x += (data->a[count].z - z)*data->b[count].y - (data->a[count].y - y)*data->b[count].z;
    	Mg_y += (data->a[count].y - y)*data->b[count].x - (data->a[count].x - x)*data->b[count].y;
    	Mg_z += (data->a[count].x - x)*data->b[count].z - (data->a[count].z - z)*data->b[count].x;
    }
    //printf("%dPg=%f\n",Np,data->b[Np-100].y);
    Pg = sqrt(pow(Pg_x,2) + pow(Pg_y,2) + pow(Pg_z,2));
    Mg = sqrt(pow(Mg_x,2) + pow(Mg_y,2) + pow(Mg_z,2));

    Pg /= Np;
    Mg /= Np;
    glColor3f(1.0, 0.0, 0.0);
    glBegin(GL_LINE_LOOP);
    glVertex3f(-WORLD_SIZE,0,-WORLD_SIZE);

    for(int count=0;count<1000-1;count++){
    	Pg_tmp[count] = Pg_tmp[count+1];
    	glVertex3f((count/1000.0 -0.5) *2*WORLD_SIZE,50*Pg_tmp[count],-WORLD_SIZE);
    }
    Pg_tmp[Na-1] = Pg;
    glVertex3f((1000/1000.0 -0.5) *2*WORLD_SIZE,50*Pg,-WORLD_SIZE);

    glVertex3f(WORLD_SIZE,Pg,-WORLD_SIZE);
    glClearColor((float)(156.0/255.0),(float)(167.0/255.0),(float)(186.0/255.0),0);
    glVertex3f(WORLD_SIZE,0,-WORLD_SIZE);
    glEnd();

    glColor3f(0.0, 0.0, 1.0);
    glBegin(GL_LINE_LOOP);
    glVertex3f(-WORLD_SIZE,0,-WORLD_SIZE);

    for(int count=0;count<1000-1;count++){
    	Mg_tmp[count] = Mg_tmp[count+1];
    	glVertex3f((count/1000.0 -0.5) *2*WORLD_SIZE,50*Mg_tmp[count],-WORLD_SIZE);
    }
    Mg_tmp[Na-1] = Mg;
    glVertex3f((1000/1000.0 -0.5) *2*WORLD_SIZE,50*Mg,-WORLD_SIZE);

    glVertex3f(WORLD_SIZE,Mg,-WORLD_SIZE);
    glClearColor((float)(156.0/255.0),(float)(167.0/255.0),(float)(186.0/255.0),0);
    glVertex3f(WORLD_SIZE,0,-WORLD_SIZE);
    glEnd();

    glColor3f(0.0, 0.0, 0.0);
    glBegin(GL_LINE_LOOP);
    glVertex3f(-WORLD_SIZE,0,-WORLD_SIZE);
    glVertex3f(WORLD_SIZE,0,-WORLD_SIZE);
    glEnd();*/

    //render cube
    glColor3f(0.0, 0.0, 0.0);
    glBegin(GL_LINE_LOOP);
    float size = FIELD_SIZE;
    glVertex3d(-size,-size,-size);
    glVertex3d(size,-size,-size);
    glVertex3d(size,size,-size);
    glVertex3d(-size,size,-size);
    glVertex3d(-size,-size,-size);
    glVertex3d(-size,-size,size);
    glVertex3d(size,-size,size);
    glVertex3d(size,-size,-size);
    glVertex3d(-size,-size,-size);
    glVertex3d(-size,-size,size);
    glVertex3d(-size,size,size);
    glVertex3d(-size,size,-size);
    glVertex3d(-size,-size,-size);
    glVertex3d(-size,-size,size);
    glVertex3d(size,-size,size);
    glVertex3d(size,size,size);
    glVertex3d(size,size,-size);
    glVertex3d(size,size,size);
    glVertex3d(-size,size,size);
    glVertex3d(-size,-size,size);
    glEnd();

    //render character]
    glColor3f(0.0, 0.0, 0.0);
    char str1[128];
    glRasterPos3d(size,size,size);
    sprintf(str1,"(%f,%f,%f)",size,size,size);
    glutBitmapString(GLUT_BITMAP_HELVETICA_12,(const unsigned char*)(str1));
    glRasterPos3d(-size,size,size);
    sprintf(str1,"(%f,%f,%f)",-size,size,size);
    glutBitmapString(GLUT_BITMAP_HELVETICA_12,(const unsigned char*)(str1));
    glRasterPos3d(size,-size,size);
    sprintf(str1,"(%f,%f,%f)",size,-size,size);
    glutBitmapString(GLUT_BITMAP_HELVETICA_12,(const unsigned char*)(str1));
    glRasterPos3d(size,size,-size);
    sprintf(str1,"(%f,%f,%f)",size,size,-size);
    glutBitmapString(GLUT_BITMAP_HELVETICA_12,(const unsigned char*)(str1));
    glRasterPos3d(-size,-size,size);
    sprintf(str1,"(%f,%f,%f)",-size,-size,size);
    glutBitmapString(GLUT_BITMAP_HELVETICA_12,(const unsigned char*)(str1));
    glRasterPos3d(-size,size,-size);
    sprintf(str1,"(%f,%f,%f)",-size,size,-size);
    glutBitmapString(GLUT_BITMAP_HELVETICA_12,(const unsigned char*)(str1));
    glRasterPos3d(size,-size,-size);
    sprintf(str1,"(%f,%f,%f)",size,-size,-size);
    glutBitmapString(GLUT_BITMAP_HELVETICA_12,(const unsigned char*)(str1));
    glRasterPos3d(-size,-size,-size);
    sprintf(str1,"(%f,%f,%f)",-size,-size,-size);
    glutBitmapString(GLUT_BITMAP_HELVETICA_12,(const unsigned char*)(str1));


    glRasterPos3d(size/2.0,size,size);
    sprintf(str1,"step = %ld\n",data->time);
    glutBitmapString(GLUT_BITMAP_HELVETICA_12,(const unsigned char*)(str1));


    glPointSize(2+point_size);
    glColor3f(0.0, 0.0, 0.0);
#if DEBUG == 1
    glPointSize(3);
    glColor3f(1.0, 0.0, 0.0);
#endif
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
#if DRAW_CUDA == 1

    computeFPS();
#endif

}

void timerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}
#if DRAW_CUDA == 1

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	//printf("ok\n");
    switch (key)
    {
        case (27) :
            exit(EXIT_SUCCESS);
            break;
            //fclose(fp);
            //printf("ok2\n");
        case 'u':
        	printf("perspective=%f\n",perspective);
        	perspective += 0.1;
        	break;
        case 'd':
        	perspective -= 0.1;
        	break;
        case 'i':
        	printf("point size=%f\n",point_size);
        	point_size += 0.1;
        	break;
        case 'f':
        	point_size -= 0.1;
        	break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

#if DRAW_CUDA == 1

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda(int argc, char **argv, const GLuint &vbo)
{
    if (!d_vbo_buffer)
    {
        cudaGraphicsUnregisterResource(cuda_vbo_resource);

        // map buffer object
        glBindBuffer(GL_ARRAY_BUFFER_ARB, vbo);
        float *data = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

        // check result
        if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
        {
            // write file for regression test
            sdkWriteFile<float>("./data/regression.dat",
                                data, mesh_width * mesh_height * 3, 0.0, false);
        }

        // unmap GL buffer object
        if (!glUnmapBuffer(GL_ARRAY_BUFFER))
        {
            fprintf(stderr, "Unmap buffer failed.\n");
            fflush(stderr);
        }

        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
                                                     cudaGraphicsMapFlagsWriteDiscard));

        SDK_CHECK_ERROR_GL();
    }
}
#endif










///////////////////////////////////
// for drawing controlled by cpu //
///////////////////////////////////

void init(void) {
	glClearColor(1.0, 1.0, 1.0, 1.0);
}

void resize(int w, int h)
{
  /* ウィンドウ全体をビューポートにする */
  glViewport(0, 0, w, h);

  /* 変換行列の初期化 */
  glLoadIdentity();

  /* スクリーン上の表示領域をビューポートの大きさに比例させる */
  glOrtho(-w / 200.0, w / 200.0, -h / 200.0, h / 200.0, -1.0, 1.0);
}

int draw_in(int argc, char *argv[])
{
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA);
  glutCreateWindow(argv[0]);
  glutDisplayFunc(display);
  glutReshapeFunc(resize);
  glutMouseFunc(mouse);
  glutKeyboardFunc(keyboard);
  init();
  glutMainLoop();
  return 0;
}

void* draw( void *pvoidData ){

    data = (DataStruct*)pvoidData;

    pArgc = data->pArgc;
    pArgv = data->pArgv;

    draw_in(pArgc[0], pArgv);

	return 0;
}





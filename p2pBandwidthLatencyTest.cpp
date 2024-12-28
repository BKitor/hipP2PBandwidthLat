#include <cstdio>
#include <hip/amd_detail/amd_hip_runtime_pt_api.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <vector>

#include "helper_timer.h"

using namespace std;

typedef enum {
  P2P_WRITE = 0,
  P2P_READ = 1,
} P2PDataTransfer;

typedef enum {
  CE = 0, // copy engine
  SM = 1, // streaming-multiprocessors
} P2PEngine;

P2PEngine p2p_mechanism = CE; 
                               

#define hipCheckError() {                                             \
  hipError_t lhipe = hipGetLastError();                               \
  if (hipe != hipSuccess || lhipe != hipSuccess) {                    \
    fprintf(stderr, "HIP Failure %s:%d: '%s'\n", __FILE__, __LINE__,  \
        hipGetErrorString(hipe));                                     \
    exit(-1);                                                         \
  }                                                                   \
} 

__global__ void delay(volatile int *flag, 
                      unsigned long long timeout_clocks = 10000000) {
  long long int start_clock, sample_clock;
  start_clock = clock64();

  while (!*flag) {
    sample_clock = clock64();
    if (sample_clock - start_clock > timeout_clocks) {
      break;
    }
  }
}

__global__ void copyp2p(int4 *__restrict__ dest, int4 const *__restrict__ src, 
                        size_t num_elems) {
  size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t gridSize = blockDim.x * gridDim.x;

#pragma unroll 5
  for (size_t i = globalId; i < num_elems; i+= gridSize) {
    dest[i] = src[i];
  }
}

void checkP2PAccess(int numGPUs) {
  enum hipError_t hipe = hipSuccess;
  vector<int> can_access_vec;
  for (int i = 0; i < numGPUs; i++) {
    hipe = hipSetDevice(i);
    hipCheckError();
    for (int j = 0; j < numGPUs; j++) {
      int access;
      if (i == j)
        continue;

      hipe = hipDeviceCanAccessPeer(&access, i, j);
      if (access)
        can_access_vec.push_back(j);
    }
    printf("Device=%d can access [", i);
    for (const int& i: can_access_vec) 
      printf("%d, ", i);
    printf("]\n");
    can_access_vec.clear();
  }
}

void performP2PCopy(int *dest, int destDevice, int *src, int srcDevice, 
                    int num_elems, int repeat, bool p2paccess, hipStream_t streamToRun) {
  hipError_t hipe;
  int blockSize = 0;
  int numBlocks = 0;
  
  hipe = hipOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, copyp2p);
  hipCheckError();
  
  if (p2p_mechanism == SM && p2paccess) {
    for (int r = 0; r < repeat; r++) 
      copyp2p<<<numBlocks, blockSize, 0, streamToRun>>>(
          (int4 *)dest, (int4 *)src, num_elems / 4);
  } else {
    for (int r = 0; r < repeat; r++) 
      hipe = hipMemcpyPeerAsync(dest, destDevice, src, srcDevice, 
                         sizeof(int) * num_elems, streamToRun);
  }
}

void outputBidirectionalBandwidthMatrix(int numElems, int numGPUs, bool p2p) {
  hipError_t hipe;
  int repeat = 5;
  volatile int *flag = NULL;
  vector<int *> buffers(numGPUs);
  vector<int *> buffersD2D(numGPUs);
  vector<hipEvent_t> start(numGPUs);
  vector<hipEvent_t> stop(numGPUs);
  vector<hipStream_t> stream0(numGPUs);
  vector<hipStream_t> stream1(numGPUs);

  hipe = hipHostAlloc((void **)&flag, sizeof(*flag), hipHostAllocPortable);
  hipCheckError();

  for (int d = 0; d < numGPUs; d++) {
    hipe = hipSetDevice(d);
    hipe = hipMalloc(&buffers[d], numElems * sizeof(int));
    hipe = hipMemset(buffers[d], 0, numElems * sizeof(int));
    hipe = hipMalloc(&buffersD2D[d], numElems * sizeof(int));
    hipe = hipMemset(buffersD2D[d], 0, numElems * sizeof(int));
    hipCheckError();
    hipe = hipEventCreate(&start[d]);
    hipCheckError();
    hipe = hipEventCreate(&stop[d]);
    hipCheckError();
    hipe = hipStreamCreateWithFlags(&stream0[d], hipStreamNonBlocking);
    hipCheckError();
    hipe = hipStreamCreateWithFlags(&stream1[d], hipStreamNonBlocking);
    hipCheckError();
  }

  vector<double> bandwidthMatrix(numGPUs * numGPUs);

  for (int i = 0; i < numGPUs; i++) {
    hipe = hipSetDevice(i);

    for (int j = 0; j < numGPUs; j++) {
      int access = 0;
      if (p2p) {
        hipe = hipDeviceCanAccessPeer(&access, i, j);
        if (access) {
          hipe = hipSetDevice(i);
          hipe = hipDeviceEnablePeerAccess(j, 0);
          hipe = hipSetDevice(j);
          hipe = hipDeviceEnablePeerAccess(i, 0);
          hipCheckError();
        }
      }


      hipe = hipSetDevice(i);
      hipe = hipStreamSynchronize(stream0[i]);
      hipe = hipStreamSynchronize(stream1[i]);
      hipCheckError();

      *flag = 0;
      hipe = hipSetDevice(i);
      delay<<<1, 1, 0, stream0[i]>>>(flag);
      hipCheckError();

      hipe = hipEventRecord(start[i], stream0[i]);
      hipe = hipStreamWaitEvent(stream1[j], start[i], 0);

      if (i == j) {
        performP2PCopy(buffers[i], i, buffersD2D[i], i, numElems, repeat,
                       access, stream0[i]);
        performP2PCopy(buffersD2D[i], i, buffers[i], i, numElems, repeat,
                       access, stream1[i]);
      } else {
        if (access && p2p_mechanism == SM) 
          hipe = hipSetDevice(j);
        performP2PCopy(buffers[i], i, buffers[j], j, numElems, repeat, access,
                       stream1[j]);
        if (access && p2p_mechanism == SM)
          hipe = hipSetDevice(i);
        performP2PCopy(buffers[j], j, buffers[i], i, numElems, repeat, access, 
                       stream0[i]);
      }

      hipe = hipEventRecord(stop[j], stream1[j]);
      hipe = hipStreamWaitEvent(stream0[i], stop[j], 0);
      hipe = hipEventRecord(stop[i], stream0[i]);

      *flag = 1;
      hipe = hipStreamSynchronize(stream0[i]);
      hipe = hipStreamSynchronize(stream1[j]);
      hipCheckError();

      float time_ms;
      hipe = hipEventElapsedTime(&time_ms, start[i], stop[i]);
      double time_s = time_ms / 1e3;

      double gb = 2.0 * numElems * sizeof(int) * repeat / (double)1e9;
      if (i == j) {
        gb *= 2;
      }
      bandwidthMatrix[i * numGPUs + j] = gb / time_s;
      if (p2p && access) {
        hipe = hipSetDevice(i);
        hipe = hipDeviceDisablePeerAccess(j);
        hipe = hipSetDevice(j);
        hipe = hipDeviceDisablePeerAccess(i);
        hipCheckError();
      }
    }
    printf(".");
    fflush(stdout);
  }

  printf("\n");

  printf("  D\\D");

  for (int j = 0; j < numGPUs; j++)
    printf("%6d ", j);

  printf("\n");

  for (int i = 0; i < numGPUs; i++) {
    printf("%6d ", i);

    for (int j = 0; j < numGPUs; j++)
      printf("%6.02f ", bandwidthMatrix[i * numGPUs + j]);

    printf("\n");
  }

  for (int d = 0; d < numGPUs; d++) {
    hipe = hipSetDevice(d);
    hipe = hipFree(buffers[d]);
    hipe = hipFree(buffersD2D[d]);
    hipCheckError();
    hipe = hipEventDestroy(start[d]);
    hipCheckError();
    hipe = hipEventDestroy(stop[d]);
    hipCheckError();
    hipe = hipStreamDestroy(stream0[d]);
    hipCheckError();
    hipe = hipStreamDestroy(stream1[d]);
    hipCheckError();
  }

  hipe = hipFreeHost((void *)flag);
  hipCheckError();
}

void outputLatencyMatrix(int numGPUs, bool p2p, P2PDataTransfer p2p_method) {
  hipError_t  hipe;
  int repeat = 100;
  int numElems = 4;
  volatile int *flag = NULL;
  StopWatchInterface *stopWatch = NULL;
  vector<int *> buffers(numGPUs);
  vector<int *> buffersD2D(numGPUs);
  vector<hipStream_t> stream(numGPUs);
  vector<hipEvent_t> start(numGPUs);
  vector<hipEvent_t> stop(numGPUs);

  hipe = hipHostAlloc((void **)&flag, sizeof(*flag), hipHostAllocPortable);
  hipCheckError();

  if (!sdkCreateTimer(&stopWatch)) {
    printf("Failed to create stop watch\n");
    exit(-1);
  }
  sdkStartTimer(&stopWatch);


  for (int d = 0; d < numGPUs; d++) {
    hipe = hipSetDevice(d);
    hipe = hipStreamCreateWithFlags(&stream[d], hipStreamNonBlocking);
    hipe = hipMalloc(&buffers[d], sizeof(int) * numElems);
    hipe = hipMemset(buffers[d], 0, sizeof(int) * numElems);
    hipe = hipMalloc(&buffersD2D[d], sizeof(int) * numElems);
    hipe = hipMemset(buffersD2D[d], 0, sizeof(int) * numElems);
    hipCheckError();
    hipe = hipEventCreate(&start[d]);
    hipCheckError();
    hipe = hipEventCreate(&stop[d]);
    hipCheckError();
  }

  vector<double> gpuLatencyMatrix(numGPUs * numGPUs);
  /*vector<double> cpuLatencyMatrix(numGPUs * numGPUs);*/

  for (int i = 0; i < numGPUs; i++) {
    hipe = hipSetDevice(i);
    for (int j = 0; j < numGPUs; j++) {
      int access = 0;
      if (p2p) {
        hipe = hipDeviceCanAccessPeer(&access, i, j);
        if (access) {
          hipe = hipDeviceEnablePeerAccess(j, 0);
          hipCheckError();
          hipe = hipSetDevice(j);
          hipe = hipDeviceEnablePeerAccess(i, 0);
          hipe = hipSetDevice(i);
          hipCheckError();
        }
      }
      hipe = hipStreamSynchronize(stream[i]);
      hipCheckError();

      *flag = 0;
      delay<<<1, 1, 0, stream[i]>>>(flag);
      hipCheckError();
      hipe = hipEventRecord(start[i], stream[i]);

      sdkResetTimer(&stopWatch);
      if (i == j) {
        performP2PCopy(buffers[i], i, buffersD2D[i], i, numElems, repeat, access, stream[i]);
      } else {
        if (p2p_method == P2P_WRITE) {
          performP2PCopy(buffers[j], j, buffers[i], i, numElems, repeat, access, stream[i]);
        } else {
          performP2PCopy(buffers[i], i, buffers[j], j, numElems, repeat, access, stream[i]);
        }
      }
      /*float cpu_time_ms = sdkGetTimerValue(&stopWatch);*/

      hipe = hipEventRecord(stop[i], stream[i]);
      *flag = 1;
      hipe = hipStreamSynchronize(stream[i]);
      hipCheckError();

      float gpu_time_ms;
      hipe = hipEventElapsedTime(&gpu_time_ms, start[i], stop[i]);
      gpuLatencyMatrix[i * numGPUs + j] = gpu_time_ms * 1e3 / repeat;
      /*cpuLatencyMatrix[i * numGPUs + j] = cpu_time_ms * 1e3 / repeat;*/
      if (p2p && access) {
        hipe = hipDeviceDisablePeerAccess(j);
        hipe = hipSetDevice(j);
        hipe = hipDeviceDisablePeerAccess(i);
        hipe = hipSetDevice(i);
        hipCheckError();
      }
    }
    printf(".");
    fflush(stdout);
  }

  printf("\n GPU");
  for (int j = 0; j < numGPUs; j++)
    printf("%6d ", j);
  printf("\n");
  for (int i = 0; i < numGPUs; i++) {
    printf("%6d ", i);
    for (int j = 0; j < numGPUs; j++)
      printf("%6.02f ", gpuLatencyMatrix[i * numGPUs + j]);
    printf("\n");
  }

  /*printf("\n CPU");*/
  /*for (int j = 0; j < numGPUs; j++)*/
  /*  printf("%6d ", j);*/
  /*printf("\n");*/
  /*for (int i = 0; i < numGPUs; i++) {*/
  /*  printf("%6d ", i);*/
  /*  for (int j = 0; j < numGPUs; j++)*/
  /*    printf("%6.02f ", cpuLatencyMatrix[i * numGPUs + j]);*/
  /*  printf("\n");*/
  /*}*/

  for (int d = 0; d < numGPUs; d++) {
    hipe = hipSetDevice(d);
    hipe = hipFree(buffers[d]);
    hipe = hipFree(buffersD2D[d]);
    hipCheckError();
    hipe = hipEventDestroy(start[d]);
    hipCheckError();
    hipe = hipEventDestroy(stop[d]);
    hipCheckError();
    hipe = hipStreamDestroy(stream[d]);
    hipCheckError();
  }

  sdkDeleteTimer(&stopWatch);

  hipe = hipFreeHost((void *)flag);
  hipCheckError()
}

int main(int argc, char **argv) {
  enum hipError_t hipe = hipSuccess;
  int numGPUs, numElems = 40000000;
  P2PDataTransfer p2p_method = P2P_WRITE;

  hipe = hipGetDeviceCount(&numGPUs);
  hipCheckError();

  // TODO: cmdline w/ getopt or something

  printf("BK HIP P2P Latency Test; ndevs: %d \n", numGPUs );

  for (int i = 0; i<numGPUs; i++) {
    hipDeviceProp_t prop;
    hipe = hipGetDeviceProperties(&prop, i);
    hipCheckError();
    printf("Device: %d, %s, pciBusID: %x, pciDeviceID: %x, pciDomainID:%x\n", i,
           prop.name, prop.pciBusID, prop.pciDeviceID, prop.pciDomainID);
  }

  /*checkP2PAccess(numGPUs);*/

  printf("P2P Conn Matrix\n");
  printf("  D\\D");

  for (int j = 0; j < numGPUs; j++) {
    printf("%6d", j);
  }
  printf("\n");

  for (int i = 0; i< numGPUs; i++) {
    printf("%6d\t", i);
    for (int j = 0; j < numGPUs; j++) {
      if (i != j) {
        int access;
        hipe = hipDeviceCanAccessPeer(&access, i, j);
        hipCheckError();
        printf("%6d", (access) ? 1: 0);
      } else {
        printf("%6d", 1);
      }
    }
    printf("\n");
  }

  /*outputBandwidthMatrix();*/
  printf("Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)\n");
  outputBidirectionalBandwidthMatrix(numElems, numGPUs, false);
  printf("Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)\n");
  outputBidirectionalBandwidthMatrix(numElems, numGPUs, true);

  printf("P2P=Disabled Latency Matrix (us)\n");
  outputLatencyMatrix(numGPUs, false, P2P_WRITE);
  printf("P2P=Enabled Latency (P2P Writes) Matrix (us)\n");
  outputLatencyMatrix(numGPUs, true, P2P_WRITE);
  if (p2p_method == P2P_READ) {
    printf("P2P=Enabled Latency (P2P Reads) Matrix (us)\n");
    outputLatencyMatrix(numGPUs, true, p2p_method);
  }

} 

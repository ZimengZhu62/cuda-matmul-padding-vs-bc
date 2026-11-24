#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
//A=M*N,B=N*K
__global__ void matmul(float *da,float *db,float *dc,int M,int N,int K){
    __shared__ float A[16][16];
    __shared__ float B[16][16];
    int row=blockIdx.y*16+threadIdx.y;
    int col=blockIdx.x*16+threadIdx.x;
    int circunum=(N+15)/16;
    float sum=0;
    int remainder=N-16*circunum+16;
    for(int i=0;i<circunum;i++){

        A[threadIdx.y][threadIdx.x]=da[row*N+i*16+threadIdx.x];
        B[threadIdx.y][threadIdx.x]=db[K*(threadIdx.y+i*16)+col];

        __syncthreads();

        for(int j=0;j<16;j++){
            sum+=A[threadIdx.y][j]*B[j][threadIdx.x];
        }

        __syncthreads();
    }
    dc[row*K+col]=sum;
}
int main(){
    int M,N,K;
    printf("输入 M N K: ");
    scanf("%d %d %d", &M, &N, &K);
    int Mplus=((M+15)/16)*16;
    int Nplus=((N+15)/16)*16;
    int Kplus=((K+15)/16)*16;
    dim3 gridsize(Kplus/16,Mplus/16);
    dim3 blocksize(16,16);
    float *ha,*hb,*hcfake,*hc;
    size_t sizea=sizeof(float)*Mplus*Nplus;
    size_t sizeb=sizeof(float)*Kplus*Nplus;
    size_t sizecfake=sizeof(float)*Mplus*Kplus;
    size_t sizec=sizeof(float)*M*K;
    ha=(float*)malloc(sizea);
    hb=(float*)malloc(sizeb);
    hcfake=(float*)malloc(sizecfake);
    hc=(float*)malloc(sizec);
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            scanf("%f",&ha[i*Nplus+j]);
        }
        for(int j=N;j<Nplus;j++){
            ha[i*Nplus+j]=0;
        }
    }
    for(int i=M;i<Mplus;i++){
        for(int j=0;j<Nplus;j++){
            ha[i*Nplus+j]=0;
        }
    }
    for(int i=0;i<N;i++){
        for(int j=0;j<K;j++){
            scanf("%f",&hb[i*Kplus+j]);
        }
        for(int j=K;j<Kplus;j++){
            hb[i*Kplus+j]=0;
        }
    }
    for(int i=N;i<Nplus;i++){
        for(int j=0;j<Kplus;j++){
            hb[i*Kplus+j]=0;
        }

    }

    float *da,*db,*dc;
    cudaMalloc((void**)&da,sizea);
    cudaMalloc((void**)&db,sizeb);
    cudaMalloc((void**)&dc,sizecfake);
    cudaMemcpy(da,ha,sizea,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,sizeb,cudaMemcpyHostToDevice);
    matmul<<<gridsize,blocksize>>>(da,db,dc,Mplus,Nplus,Kplus);
    cudaMemcpy(hcfake,dc,sizecfake,cudaMemcpyDeviceToHost);
    int lambda=0;
    for(int i=0;i<M;i++){
        for(int j=0;j<K;j++){
            hc[lambda]=hcfake[i*Kplus+j];
            lambda++;
        }
    }

    free(ha);
    free(hb);
    free(hc);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;

}
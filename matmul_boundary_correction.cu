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
    for(int i=0;i<circunum-1;i++){
        if(row<M&&col<K){
            A[threadIdx.y][threadIdx.x]=da[row*N+i*16+threadIdx.x];
            B[threadIdx.y][threadIdx.x]=db[K*(threadIdx.y+i*16)+col];

        }
        else{
            A[threadIdx.y][threadIdx.x]=0;
            B[threadIdx.y][threadIdx.x]=0;

        }
        __syncthreads();

        for(int j=0;j<16;j++){
            sum+=A[threadIdx.y][j]*B[j][threadIdx.x];
        }

        __syncthreads();
    }
    if(row<M&&col<K){
        if(threadIdx.x<remainder&&threadIdx.y<remainder){
            A[threadIdx.y][threadIdx.x]=da[row*N+(circunum-1)*16+threadIdx.x];
            B[threadIdx.y][threadIdx.x]=db[K*(threadIdx.y+(circunum-1)*16)+col];
        }
        else if(threadIdx.x<remainder&&threadIdx.y>=remainder){
            A[threadIdx.y][threadIdx.x]=da[row*N+(circunum-1)*16+threadIdx.x];
            B[threadIdx.y][threadIdx.x]=0;
        }
        else if(threadIdx.x>=remainder&&threadIdx.y<remainder){
            A[threadIdx.y][threadIdx.x]=0;
            B[threadIdx.y][threadIdx.x]=db[K*(threadIdx.y+(circunum-1)*16)+col];
        }
        else{
            A[threadIdx.y][threadIdx.x]=0;
            B[threadIdx.y][threadIdx.x]=0;
        }
    }
    else{
        A[threadIdx.y][threadIdx.x]=0;
        B[threadIdx.y][threadIdx.x]=0;
    }
    __syncthreads();

    for(int j=0;j<(remainder);j++){
        sum+=A[threadIdx.y][j]*B[j][threadIdx.x];
    }

    __syncthreads();
    if(row<M&&col<K){
        dc[row*K+col]=sum;
    }
}
int main(){
    int M,N,K;
    printf("输入 M N K: ");
    scanf("%d %d %d", &M, &N, &K);
    dim3 gridsize((K+15)/16,(M+15)/16);
    dim3 blocksize(16,16);
    float *ha,*hb,*hc;
    size_t sizea=sizeof(float)*M*N;
    size_t sizeb=sizeof(float)*K*N;
    size_t sizec=sizeof(float)*M*K;
    ha=(float*)malloc(sizea);
    hb=(float*)malloc(sizeb);
    hc=(float*)malloc(sizec);
    for(int i=0;i<M*N;i++){
        scanf("%f",&ha[i]);
    }
    for(int i=0;i<N*K;i++){
        scanf("%f",&hb[i]);
    }
    float *da,*db,*dc;
    cudaMalloc((void**)&da,sizea);
    cudaMalloc((void**)&db,sizeb);
    cudaMalloc((void**)&dc,sizec);
    cudaMemcpy(da,ha,sizea,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,sizeb,cudaMemcpyHostToDevice);
    matmul<<<gridsize,blocksize>>>(da,db,dc,M,N,K);
    cudaMemcpy(hc,dc,sizec,cudaMemcpyDeviceToHost);
    free(ha);
    free(hb);
    free(hc);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;

}
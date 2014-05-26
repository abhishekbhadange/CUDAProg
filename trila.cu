#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

struct point
{
	float x;
	float y;
};	
struct dist
{
	float da;
	float db;
	float dc;
};

float eucli(float fx, float fy)
{
	return sqrt(fx * fx + fy * fy);
}

__global__ void trilaterate(struct point a, struct point b, struct point c, struct dist *d_set, struct point *d_trail, int NUM)
{
	float a1Sq = a.x * a.x, a2Sq = b.x * b.x, a3Sq = c.x * c.x, b1Sq = a.y * a.y, b2Sq = b.y * b.y, b3Sq = c.y * c.y;
	float r1Sq, r2Sq, r3Sq, denom1, numer1, denom2, numer2;
	float a1 = a.x, a2 = b.x, a3 = c.x, b1 = a.y, b2 = b.y, b3 = c.y; 
	int i;
	
	for(i=0; i < NUM; i++)
	{
			r1Sq = d_set[i].da * d_set[i].da;
			r2Sq = d_set[i].db * d_set[i].db;
			r3Sq = d_set[i].dc * d_set[i].dc;
			
			numer1 = (a2 - a1) * (a3Sq + b3Sq - r3Sq) + (a1 - a3) * (a2Sq + b2Sq - r2Sq) + (a3 - a2) * (a1Sq + b1Sq - r1Sq);
			denom1 = 2 * (b3 * (a2 - a1) + b2 * (a1 - a3) + b1 * (a3 - a2));
			d_trail[i].y = numer1/denom1;
			
			numer2 = r2Sq - r1Sq + a1Sq - a2Sq + b1Sq - b2Sq - 2 * (b1 - b2) * d_trail[i].y;
			denom2 = 2 * (a1 - a2);
			d_trail[i].x = numer2/denom2;
	}
}

int main(int argc, char *argv[])
{
	cudaEvent_t start, stop;
	float etime;
	int i, j=0;
	float fx, fy, gx, gy, z = 5.0;
	
	int NUM;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	if (argc != 2) 
	{
	  printf("Check you arguments!\n");
	  exit(1);
    }
	
	struct point a, b, c;
	
	a.x = 1.67; a.y = 2.58;
	b.x = 3.74; b.y = 2.08;
	c.x = 5.12; c.y = 3.95;
	
	struct point init;
	init.x = 3.12;
	init.y = 4.27;
	
	NUM = atoi(argv[1]);
	
	struct point trail[NUM], avg_trail[(NUM/4)], ret_avg_trail[(NUM/4)];
	struct point *d_trail, *h_trail;
	
	trail[0] = init;
	
	srand(time(NULL));
	
	for(i=1; i<NUM; i++)
	{
		gx = ((float)rand()/(float)(RAND_MAX)) * z;
		gx = floorf(gx * 100) / 100;
		gy = ((float)rand()/(float)(RAND_MAX)) * z;
		gy = floorf(gy * 100) / 100;
		trail[i].x = (floorf(trail[i-1].x * 100 + 0.5) / 100) + gx;
		trail[i].y = (floorf(trail[i-1].y * 100 + 0.5) / 100) + gy;	
	}
	
	for(i=0; i<(NUM/4); i++)
	{
		avg_trail[i].x = (trail[j].x + trail[j+1].x + trail[j+2].x + trail[j+3].x) / 4;
		avg_trail[i].y = (trail[j].y + trail[j+1].y + trail[j+2].y + trail[j+3].y) / 4;
		j += 4;
	}
	
	printf("\nAvg. Random Trail at Host\n");
	for(i=0; i<(NUM/4); i++)
	{
		printf("(%f, %f)\n", avg_trail[i].x, avg_trail[i].y);
	}
	
	struct dist *set;
	
	size_t size = NUM * sizeof(struct dist);
	set = (struct dist *)malloc(size);
	
	size_t sz = NUM * sizeof(struct point);
	h_trail = (struct point *)malloc(sz);
	
	for(i=0; i<NUM; i++)
	{
		fx = trail[i].x - a.x;
		fy = trail[i].y - a.y;
		set[i].da = eucli(fx, fy);
		fx = trail[i].x - b.x;
		fy = trail[i].y - b.y;
		set[i].db = eucli(fx, fy);
		fx = trail[i].x - c.x;
		fy = trail[i].y - c.y;
		set[i].dc = eucli(fx, fy);
	}
	
	struct dist *d_set;
	cudaMalloc((void **) &d_set, size);
	
	cudaMalloc((void **) &d_trail, sz);
	
	cudaMemcpy(d_set, set, sizeof(struct dist)*NUM, cudaMemcpyHostToDevice);
	
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
    int nBlocks = devProp.multiProcessorCount;
	int blockSize = devProp.warpSize;	

	printf("\nU: %d\n", nBlocks);
	printf("\nV: %d\n", blockSize);

	trilaterate <<< nBlocks, blockSize >>> (a, b, c, d_set, d_trail, NUM);
	
	cudaMemcpy(h_trail, d_trail, sizeof(struct point)*NUM, cudaMemcpyDeviceToHost);
	
	j=0;
	for(i=0; i<(NUM/4); i++)
	{
		ret_avg_trail[i].x = (h_trail[j].x + h_trail[j+1].x + h_trail[j+2].x + h_trail[j+3].x) / 4;
		ret_avg_trail[i].y = (h_trail[j].y + h_trail[j+1].y + h_trail[j+2].y + h_trail[j+3].y) / 4;
		j += 4;
	}
	
	printf("\nAvg. Generated Trail at Device\n");
	for(i=0; i<(NUM/4); i++)
	{
		printf("(%f, %f)\n", ret_avg_trail[i].x, ret_avg_trail[i].y);
	}
	printf("\n");
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&etime, start, stop);
	printf("Time elapsed: %f ms\n", etime);

    cudaEventDestroy(start);
	cudaEventDestroy(stop);
	free(set); 
	cudaFree(d_set);
	cudaFree(d_trail);
	cudaFree(h_trail);
	
	return 0;
}

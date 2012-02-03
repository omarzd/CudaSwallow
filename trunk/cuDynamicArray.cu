
/*-------------------------------------------------------------------------.
|  Dynamic Array For Parse Tree Stored as History of Actions Taken         |
`-------------------------------------------------------------------------*/

__global__ void initializeActionsTaken ( dynamicArray *newArray )
{
	actionsTaken = newArray;
}

__global__ void initializeDeviceArray ( dynamicArray mainArray , dynamicArray *destinationArray )
{
	superCudaMemcpy( (byte*)destinationArray , (byte*)&mainArray , sizeof(dynamicArray) );
}

__device__ void addElement ( actionHistorySnippet DATA , dynamicArray *destinationArray )
{
	((unsigned int*)destinationArray->buffer)[destinationArray->index++] = *((unsigned int*)&DATA);
}

__device__ void addActionTaken ( actionHistorySnippet DATA )
{
	((unsigned int*)actionsTaken->buffer)[actionsTaken->index++] = *((unsigned int*)&DATA);
}

void initializeActionsTakenArray( int lInputSize )
{	
	cudaMalloc( (void**)&lActionsTaken , sizeof( dynamicArray ) );
	
	cudaMalloc( (void**)&(mainArray.buffer) , lInputSize * lInputSize * sizeof(actionHistorySnippet) );
	mainArray.upperLimit = lInputSize * lInputSize;
	mainArray.index = 0;
	
	initializeDeviceArray<<<1,1>>>( mainArray , lActionsTaken );
	cudaThreadSynchronize();
	initializeActionsTaken<<<1,1>>>( lActionsTaken );
	cudaThreadSynchronize();
}

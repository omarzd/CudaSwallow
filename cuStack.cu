
/*---------------------------------------------------.
|       STACK and Memory Management Functions        |
`---------------------------------------------------*/

/*-------------------------------------.
|  How to Test Stack Implementaion     |
`-------------------------------------*/
/*
__global__ void myStackTesting( byte* buffer )
{
	int i;
	int stackID;
	
	stackID = allocateStack();
	
	PUSH( stackID , (byte)'3' );
	PUSH( stackID , (byte)'o' );
	PUSH( stackID , (byte)'m' );
	PUSH( stackID , (byte)'a' );
	PUSH( stackID , (byte)'r' );
	PUSH( stackID , (byte)'z' );
	//PUSH( stack , sp , 0 , '\0' );
	
	buffer[0] = 0;
	
	for(i = 0; i < 6; i++)
	{
		buffer[i] = POP( stackID );
	}
	
	buffer[6] = ' ';
	buffer[7] = 'I';
	buffer[8] = 'D';
	buffer[9] = ' ';
	buffer[10] = '=';
	buffer[11] = ' ';
	buffer[12] = stackID + '0';
	buffer[13] = 0;
	
	deallocateStack( stackID );
}
*/

__device__ int stackMainLock = 0;

__device__ void superCudaMemcpy( byte *destination , byte *source , unsigned int len)
{
	unsigned int counter;
	
	for(counter = 0; counter < len; counter++)
		destination[counter] = source[counter];
}

__device__ byte PUSH( int segment , uint16 DATA )
{
	// if segment is not in range
	if( segment < 0 || segment >= mainStack.segments ) return PUSH_FAIL;	

	// precaution for overflow
	if(mainStack.sp[segment] - mainStack.lowerLimit[segment] < sizeof(DATA))
		return PUSH_FAIL;
	
	//Push a value on the stack for the respectful element.
	mainStack.sp[segment] = mainStack.sp[segment] - sizeof(DATA);
	*((uint16*)mainStack.sp[segment]) = DATA;
	return PUSH_SUCCESS;
}
__device__ byte PUSH( int segment , unsigned int DATA )
{
	if( segment == STACK_ID_INVALID ) return PUSH_FAIL;	
	
	//Push a value on the stack for the respectful element.
	mainStack.sp[segment] = mainStack.sp[segment] - sizeof(DATA);
	*((unsigned int*)mainStack.sp[segment]) = DATA;
	return PUSH_SUCCESS;
}

__device__ byte PUSH( int segment , byte DATA )
{
	// if segment is not in range
	if( segment < 0 || segment >= mainStack.segments ) return PUSH_FAIL;	
	
	//Push a value on the stack for the respectful element.
	mainStack.sp[segment]--;
	*(mainStack.sp[segment]) = DATA;
	return PUSH_SUCCESS;
}

__device__ byte POP( int segment )
{
	//Pop a value from the stack and return it.
	byte DATA;
	
	// if segment is not in range
	if( segment < 0 || segment >= mainStack.segments ) return 0;	
	
	DATA = *(mainStack.sp[segment]);
	mainStack.sp[segment]++;
	
	return DATA;
}

__device__ unsigned short POPshort( int segment )
{
	//Pop a value from the stack and return it.
	unsigned short DATA;
	
	// if segment is not in range
	if( segment < 0 || segment >= mainStack.segments ) return 0;	
	
	DATA = *((unsigned short*)(mainStack.sp[segment]));
	mainStack.sp[segment] = mainStack.sp[segment] + sizeof(DATA);
	
	return DATA;
}

__device__ unsigned int POPint( int segment )
{
	// Pop a value from the stack and return it.
	unsigned int DATA;
	
	// if segment is not in range
	if( segment < 0 || segment >= mainStack.segments ) return 0;	
	
	DATA = *((unsigned int*)mainStack.sp[segment]);
	mainStack.sp[segment] = mainStack.sp[segment] + sizeof(DATA);
	
	return DATA;
}

__device__ byte PEEK( int segment )
{
	// Peek at the top byte of the stack.
	byte DATA;
	
	// if segment is not in range
	if( segment < 0 || segment >= mainStack.segments ) return 0;	
	
	DATA = *(mainStack.sp[segment]);
	
	return DATA;
}

__global__ void initializeDeviceStack( stack mStack )
{
	int i;
	
	superCudaMemcpy( (byte*)&mainStack , (byte*)&mStack , sizeof(stack) );
	
	for(i = 0; i < mainStack.segments ; i++)
	{
		mainStack.sp[i] = (byte*)(mainStack.buffer + (mainStack.chunk * (i + 1)));
		mainStack.lowerLimit[i] = (byte*)(mainStack.buffer + (mainStack.chunk * i));
		mainStack.upperLimit[i] = (byte*)(mainStack.buffer + (mainStack.chunk * (i + 1)));
		mainStack.allocated[i] = STACK_SEGMENT_UNALLCOATED;
	}
}

__global__ void deinitializeDeviceStack()
{
	int i;
	
	mainStack.buffer = 0;
		
	for(i = 0; i < mainStack.segments ; i++)
	{
		mainStack.sp[i] = 0;
		mainStack.lowerLimit[i] = 0;
		mainStack.upperLimit[i] = 0;
		mainStack.allocated[i] = STACK_SEGMENT_UNALLCOATED;
	}
}

__device__ int allocateStack()
{
	int i = 0;
	
	while( atomicCAS(&stackMainLock, 0, 1) != 0 );
	
	for(i = 0; i < mainStack.segments; i++)
	{
		if(mainStack.allocated[i] == STACK_SEGMENT_UNALLCOATED)
		{
			mainStack.allocated[i] = STACK_SEGMENT_ALLCOATED;
			mainStack.sp[i] = mainStack.upperLimit[i];

			while( atomicCAS(&stackMainLock, 1, 0) != 1 );


			return i;
		}
	}
	
	while( atomicCAS(&stackMainLock, 1, 0) != 1 );
	#if DEV_DEBUG_STACK
	cuPrintf("NO STACK SPACE AVAILABLE FOR ME... I WILL DIE !!\n");
	#endif
	return STACK_ID_INVALID;
}

__device__ int allocateStack( int neededID , byte keepData )
{
	int i = 0;
	
	// if segment is not in range
	if( neededID < 0 || neededID >= mainStack.segments ) return STACK_ID_INVALID;	
	
	for(i = 0; i < mainStack.segments; i++)
	{
		if(mainStack.allocated[i] == STACK_SEGMENT_UNALLCOATED)
		{
			mainStack.allocated[i] = STACK_SEGMENT_ALLCOATED;
			if(keepData != 0)
				mainStack.sp[i] = mainStack.upperLimit[i];
			return i;
		}
	}
	
	return STACK_ID_INVALID;
}

__device__ byte copyStack( int destination , int source )
{
	// if segment is not in range
	if( destination < 0 || destination >= mainStack.segments\
	 || source      < 0 || source      >= mainStack.segments )\
	  return COPYSTACK_FAIL;	
	superCudaMemcpy( (byte*)mainStack.lowerLimit[destination] , (byte*)mainStack.lowerLimit[source] , mainStack.chunk );
	mainStack.sp[destination] = mainStack.upperLimit[destination] - (mainStack.upperLimit[source] - mainStack.sp[source]);
	return COPYSTACK_SUCCESS;
}

__device__ void deallocateStack( int stackID )
{
	// if segment is not in range
	if( stackID < 0 || stackID >= mainStack.segments ) return;	
	
	mainStack.allocated[stackID] = STACK_SEGMENT_UNALLCOATED;
	#if DEV_DEBUG_STACK
	//cuPrintf("DEALLOCATED ID %d\n", stackID);
	#endif
}

void deinitializeStack()
{
	if( mStack.buffer != 0 )
	{
		HANDLE_FREE( cudaFree, mStack.buffer );
		HANDLE_FREE( cudaFree, (byte*)mStack.sp );
		HANDLE_FREE( cudaFree, (byte*)mStack.lowerLimit );
		HANDLE_FREE( cudaFree, (byte*)mStack.upperLimit );
		HANDLE_FREE( cudaFree, mStack.allocated );
	}
	
	deinitializeDeviceStack<<<1,1>>>();
	cudaThreadSynchronize();
}

void initializeStack( int chunk , int segments )
{
// it is necessary to make sure these are freed if we're going to  \
   call initializeStack more than once in the code
	if( mStack.buffer != 0 )
	{
		HANDLE_FREE( cudaFree, mStack.buffer );
		HANDLE_FREE( cudaFree, (byte*)mStack.sp );
		HANDLE_FREE( cudaFree, (byte*)mStack.lowerLimit );
		HANDLE_FREE( cudaFree, (byte*)mStack.upperLimit );
		HANDLE_FREE( cudaFree, mStack.allocated );
	}
	
	//Allocate Stack Space
	cudaMalloc( (void **)&(mStack.buffer) , chunk * segments );
	cudaMalloc( (void **)&(mStack.sp) , segments * sizeof(byte*) );
	cudaMalloc( (void **)&(mStack.lowerLimit) , segments * sizeof(byte*) );
	cudaMalloc( (void **)&(mStack.upperLimit) , segments * sizeof(byte*) );
	cudaMalloc( (void **)&(mStack.allocated) , segments * sizeof(byte) );
	mStack.chunk = chunk;
	mStack.segments = segments;
	
	initializeDeviceStack<<<1,1>>>( mStack );
	cudaThreadSynchronize();
}

void initializeStack( stack defaultStack , int chunk , int segments )
{
	if( defaultStack.buffer != 0 )
	{
		HANDLE_FREE( cudaFree, defaultStack.buffer );
		HANDLE_FREE( cudaFree, (byte*)defaultStack.sp );
		HANDLE_FREE( cudaFree, (byte*)defaultStack.lowerLimit );
		HANDLE_FREE( cudaFree, (byte*)defaultStack.upperLimit );
	}
	
	//Allocate Stack Space
	cudaMalloc( (void **)&(defaultStack.buffer) , chunk * segments );
	cudaMalloc( (void **)&(defaultStack.sp) , segments * sizeof(byte*) );
	cudaMalloc( (void **)&(defaultStack.lowerLimit) , segments * sizeof(byte*) );
	cudaMalloc( (void **)&(defaultStack.upperLimit) , segments * sizeof(byte*) );
	defaultStack.chunk = chunk;
	defaultStack.segments = segments;
	
	initializeDeviceStack<<<1,1>>>( defaultStack );
	cudaThreadSynchronize();
}


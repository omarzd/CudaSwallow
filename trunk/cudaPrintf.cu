
	/*
	// How to use from device
	// after intialization from host
	strBuffer[0] = 0;
	it=0;		
	lenBuffer = strlen("Worker ID = ");
	//for(it=0; it<lenBuffer;it++);// ( (byte*)&("Worker ID = ") );	
	strBuffer[it++] = '0' + ((lBlockNum/1000));
	strBuffer[it++] = '0' + ((lBlockNum/100) %10);
	strBuffer[it++] = '0' + ((lBlockNum/10) %10);
	strBuffer[it++] = '0' + ((lBlockNum/1) %10);
	strBuffer[it++] = '\n';
	strBuffer[it++] = '\0';
	cudaPrintf( (byte*)&strBuffer );
	*/

/*-----------------------------------------------.
|       Debugging and Printing Facilities        |
`-----------------------------------------------*/
__device__ int printingLocked = 0;
__device__ int queue = 0;

__device__ void cudaPrintf ( byte *buffer )
{
	int i = 0;

	while( atomicCAS(&printingLocked, 0, 1) != 0 );
	
	if( debuggingDevice.index > debuggingDevice.upperLimit )
	{
		while( atomicCAS(&printingLocked, 1, 0) != 1 );
		return;
	}
	
	while((buffer[i] != 0) && (debuggingDevice.index < debuggingDevice.upperLimit))
	{
		debuggingDevice.buffer[debuggingDevice.index++] = buffer[i++];
	}
	
	while( atomicCAS(&printingLocked, 1, 0) != 1 );
}

__device__ int cudaStrLen ( byte *buffer )
{
	int i;

	for(i = 0; buffer[i] != 0; i++);
	
	return i;
}

__device__ int getStrArrayItem ( int index , byte* array , byte delimiter)
{
	int count;
	int i;
	
	i = 0;
	count = -1;

	while(count < index)
	{
		if(array[i++] == delimiter)
			count++;
	}
	
	return i;
}

__device__ int getStrArrayItem ( int index , byte* array)
{
	return getStrArrayItem ( index , array , '\0');
}

__device__ void cudaPrintf ( byte character )
{
	while( atomicCAS(&printingLocked, 0, 1) != 0 );
	
	if( debuggingDevice.index > debuggingDevice.upperLimit - (character == 0 ?  0 : 1) )
	{
		while( atomicCAS(&printingLocked, 1, 0) != 1 );
		return;
	}
	
	debuggingDevice.buffer[debuggingDevice.index++] = character;

	while( atomicCAS(&printingLocked, 1, 0) != 1 );
}

__device__ void cudaPrintf ( const char *string )
{
	cudaPrintf( ((byte*)&string) );
}

__global__ void initializeDeviceDebugging ( byte *sharedBuffer , unsigned int lInputSize )
{
	debuggingDevice.buffer = sharedBuffer;
	debuggingDevice.index = 0;
	debuggingDevice.upperLimit = lInputSize;
}

void initializeDebugging( int lInputSize )
{
	cudaMalloc( (void**)&printingBuffer , lInputSize );
	
	initializeDeviceDebugging<<<1,1>>>( printingBuffer , lInputSize);
	cudaThreadSynchronize();
}

__global__ void closeDeviceDebugging()
{
	cudaPrintf( (byte)'\0' );
}

void closeDebugging()
{
	closeDeviceDebugging<<<1,1>>>();
	cudaThreadSynchronize();
}


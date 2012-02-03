
/*--------------------------------------.
| Types and Constants Defintions		|
`--------------------------------------*/
#define PRINT_TO_FILE 1
//#define DEV_DEBUG_ANY 1   // REMEMBER: if there are cuprintfs without #if, don't disable the initialization from here
#ifndef DEV_DEBUG_ANY
 #define DEV_DEBUG_ANY (DEV_DEBUG_CLEAN||DEV_DEBUG_CRUDE||DEV_DEBUG_STACK||DEV_DEBUG_EXCEPTION)
#endif
#define DEV_DEBUG_CLEAN 0
#define DEV_DEBUG_CRUDE 1
#define DEV_DEBUG_STACK 0
#define DEV_DEBUG_EXCEPTION 1
#if PRINT_TO_FILE
  #define OUTPUT_STREAM outfile
#else 
  #define OUTPUT_STREAM stdout
#endif
#define MY_STACK_MANAGEMENT 1  // use my own stack functions.. the other approach is cudaMalloc all the way.
#define GRID_PARSE 1  // Uses a 2D grid of blocks instead of 1D
#define Perform Execute  // !!
#define MAX_ACTIONS 3 	// max num of actions per state per token
#define MAX_RHS_COUNT 3	// max element in array rhscount
#define NRULES 19  // Number of Rules including Rule 0. = YYNRULES from bison
#define ACTION_TABLE_ERROR 0
#define ACTION_TABLE_ACCEPT 127
#define MAX_N_STATES 126 // sizeof(***(table))/2	// MAX_N_STATES -- Maximum Number of States supported, 0 and 127 are reserved
#define STATE_INVALID 255   // (typeof(gotoState)) -1 -->  (byte)(-1) = 255
#define INITIAL_STATE 0
#define TERMINAL_STATE 5
#define NSTATES 24					// NSTATES -- Number of States. = YYNSTATES-1 
#define NTOKENS  7 					// NTOKENS -- Number of terminals. = YYNTOKENS
#define NNTERMS  3 					// NNTERMS -- Number of nonterminals. = YYNNTS  
#define NSYMBOLS NTOKENS+NNTERMS	// NSYMBOLS -- Number of symbols
#define N_STACK_SEGMENTS 500000 //(long long) 2*exp(1.15 * inputSize) // Maximum Number of Stack Segments needed in any single call to "Parse". This has been decided by statistics.
#define STACK_SEGMENT_SIZE 	100 // (inputSize + 1) * 2 * sizeof(int) // TODO: revert this
#define STACK_SEGMENT_UNALLCOATED 0	// assigned initially and when a segment is deallocated.
#define STACK_SEGMENT_ALLCOATED 1	// assigned when a segment is allocated.
#define STACK_ID_INVALID -1	// assigned when allocating a segment in the stack fails. also is the default value for initiated sps blocks in case they fail which they mostly do
#define STACK_FIRST_SEGMENT_ID 0
#define COPYSTACK_SUCCESS 1
#define COPYSTACK_FAIL 0
#define PUSH_SUCCESS 1
#define PUSH_FAIL 0 
#define PRINT_BUFFER_SIZE 	100 * 1024
#define SPS_ERROR 255		// undefined error
#define SPS_SUCCESS 1		// Successful
#define SPS_ACCEPT 2		// Input Accepted
#define SAFETY_PARSE_ERROR -1	    // to distinguish which kernel function caused the error
#define SAFETY_PREPROCESS_ERROR -2  // to distinguish which kernel function caused the error
#define RULEINDEX	(ruleNum+1)  // necessary for compatabiliy with bison's arrays.. not used, variable "ruleIndex" is used instead.
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#if MY_STACK_MANAGEMENT
#define PARSE_KILLSPS  { outSps[lBlockNum].statusFlag = SPS_ERROR;\
						 outSps[lBlockNum].stackID = STACK_ID_INVALID;	return;	}	
#else
#define PARSE_KILLSPS  { outSps[lBlockNum].statusFlag = SPS_ERROR;\
						 return;	}	
#endif
#define CUPRINT(a,b)   { cudaPrintfInit();\
	a,b;\
	HANDLE_ERROR( cudaDeviceSynchronize() );\
	cudaPrintfDisplay(OUTPUT_STREAM, true);\
	cudaPrintfEnd(); }
#define ASSERT(a) {if (a == NULL) { \
                       printf( "ASSERT failed in %s at line %d\n", \
                                __FILE__, __LINE__ ); \
                       exit( EXIT_FAILURE );}}

/*	
	By using HANDLE_FREE(function, pointer), you don't have to worry if
	a particular pointer has been freed before; it handles it. you can
	simply list all frees at the end of main().
*/
#define HANDLE_FREE(free,pointer) \
	do { if(pointer) free(pointer);\
		 else printf("Trying to Free NULL pointer in %s at line %d\n",\
					 __FILE__, __LINE__ );}\
	while(0);

#define PRINT(a) \
	do { fprintf(OUTPUT_STREAM, a);}\
	while(0);

typedef unsigned char byte; 
typedef unsigned short uint16;
byte *dev_inputString;
byte *dev_translatedInputString;
byte *printingBuffer;
int maxSuccessCount = 0;   // used to estimate the number of stacks needed for each iteration and print it upon a stack crash.
int maxSegmentsNeeded = -1; // used to better predict for future runs the number of stacks needed VS. inputSize
int inputSize = 0;
int mainIteration=0;
bool errorCaught=false;


struct sps // parse state
{  
	byte statusFlag;	// successful = 1, error = 255, accept = 2
	uint16 currCharIndex;
	byte stateNum;
	#if MY_STACK_MANAGEMENT
	int stackID;	 // ID of stack of unreduced tokens in the chunk
	#else
	byte *stackID;    // address of stack
	byte offset;	 // where at stack
	#endif
};

struct stack
{
	byte *buffer;	// the whole space: size to be initially allocated, i.e., once in a runtime
	byte **sp;		// array of stack pointers to every segment
	byte **lowerLimit; // array of lower limits of all segments
	byte **upperLimit; // array of upper limits of all segments
	byte *allocated;// array of flags for each segment. 0= not allocated, 1= allocated
	int  chunk;		// size of a single segment
	int  segments;  // num of segments to equally divide into chunks residing in the buffer
};

// TODO: hmmm... i need to redesign this
//This struct's size should be constrained within 4 bytes.
struct actionHistorySnippet
{
	byte blockID;
	uint16 charIndex;
	byte actionTaken;
};

struct dynamicArray
{
	actionHistorySnippet *buffer;
	int index;
	int upperLimit;
};

struct debugBuffer
{
	byte *buffer;
	unsigned int index;
	unsigned int upperLimit;
};

__device__ stack mainStack;
stack mStack;

__device__ debugBuffer debuggingDevice;

__device__ dynamicArray *actionsTaken;
dynamicArray mainArray;
dynamicArray *lActionsTaken;



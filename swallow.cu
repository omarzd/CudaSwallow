/* changes here

local stack management directive vs cudamalloc every iteration

*/

/* needed compile switches
	-arch sm_11      // for atomicCAS
	-lcudart
*/


#include <iostream>
#include <stdio.h>
#include <math.h>
#include "cuPrintf.cu"
/*--------------------------------------.
| Types and Constants Defintions		|
`--------------------------------------*/
#define PRINT_TO_FILE 0
//#define DEV_DEBUG_ANY 1   // REMEMBER: if there are cuprintfs without #if, don't disable the initialization from here
#ifndef DEV_DEBUG_ANY
 #define DEV_DEBUG_ANY (DEV_DEBUG_CLEAN||DEV_DEBUG_CRUDE||DEV_DEBUG_STACK||DEV_DEBUG_EXCEPTION)
#endif
#ifndef DEV_DEBUG_CLEAN
 #define DEV_DEBUG_CLEAN 0
#endif
#ifndef DEV_DEBUG_CRUDE
 #define DEV_DEBUG_CRUDE 1
#endif
#ifndef DEV_DEBUG_STACK
 #define DEV_DEBUG_STACK 0
#endif
#ifndef DEV_DEBUG_EXCEPTION
 #define DEV_DEBUG_EXCEPTION 1
#endif
#if PRINT_TO_FILE
  #define OUTPUT_STREAM outfile
#else 
  #define OUTPUT_STREAM stdout
#endif
#define MY_STACK_MANAGEMENT 0  // use my own stack functions.. the other approach is cudaMalloc all the way.
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
	HANDLE_ERROR( cudaThreadSynchronize() );\
	cudaPrintfDisplay(OUTPUT_STREAM, true);\
	cudaPrintfEnd(); }
#define ASSERT(a) {if (a == NULL) { \
                       printf( "ASSERT failed in %s at line %d\n", \
                                __FILE__, __LINE__ ); \
                       exit( EXIT_FAILURE );}}
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

// TODO: hmmm... i need to redesign this
//This struct's size should be constrained within 4 bytes.
struct actionHistorySnippet
{
	byte blockID;
	uint16 charIndex;
	byte actionTaken;
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

/*---------------------------.
|  Important Information.    |
`---------------------------*/

/* Limitations: data types of critical variables
	
	variable 				type			elaboration
	successCount			int
	acceptCount				int
	sps.stateNum			byte
	sps.currCharIndex		uint16			=> max input size = 65536
	table					byte***			=> max num of states = 126. very low. if going higher, u need to adjust the push and pop calls
	
*/

/* Statistics: max num of stack segments used vs input size
	n	#segments
	2	6
	3	15
	4	48
	5	174
	6	597
	7	2283
	8	7659

	In all these cases, an input of alternating between a and u has been
	entered since "auau" uses more stack segments than "aaaa" or "uuuu".

	Enter the following line in Wolfram Alpha to get a nice fitted formula for the data
	exponential fit {2,6},{3,15},{4,48},{5,174},{6,597},{7,2283},{8,7659}
	http://www1.wolframalpha.com/input/?i=curve%20fitting%20&lk=2
*/

/*	List of All mallocs; all these are being freed somewhere in the code
	By using HANDLE_FREE(function, pointer), you don't have to worry if
	a particular pointer has been freed before; it handles it. you can
	simply list all frees at the end of main().

c	cudaMalloc( (void**)&lActionsTaken , sizeof( dynamicArray ) );	
c	cudaMalloc( (void**)&(mainArray.buffer) , inputSize * inputSize * sizeof(actionHistorySnippet) );
c	youts = (char *)malloc (fileSize);	 //allocate memory for input string
c	inputString = (char *)malloc (fileSize);	 //allocate memory for input string
c	cudaMalloc( (void**)&dev_inputString, inputSize+1 );
c	cudaMalloc( (void**)&dev_translatedInputString, inputSize+1 ) );
c	cudaMalloc( (void**)&dev_successCount , sizeof(int) );
c	cudaMalloc( (void**)&dev_acceptCount , sizeof(int) );
c	cudaMalloc( (void**)&dev_spsArrIn , sizeof(sps) );
c	cudaMalloc( (void**)&dev_spsArrOut, sizeof(sps) * N ) );
c	cudaMalloc( (void**)&printingBuffer , inputSize );
c	cudaMalloc( (void **)&(mStack.buffer) , chunk * segments );
c	cudaMalloc( (void **)&(mStack.sp) , segments * sizeof(byte*) );
c	cudaMalloc( (void **)&(mStack.lowerLimit) , segments * sizeof(byte*) );
c	cudaMalloc( (void **)&(mStack.upperLimit) , segments * sizeof(byte*) );
c	cudaMalloc( (void **)&(mStack.allocated) , segments * sizeof(byte) );
c	cudaMalloc( (void **)&(defaultStack.buffer) , chunk * segments );
c	cudaMalloc( (void **)&(defaultStack.sp) , segments * sizeof(byte*) );
c	cudaMalloc( (void **)&(defaultStack.lowerLimit) , segments * sizeof(byte*) );
c	cudaMalloc( (void **)&(defaultStack.upperLimit) , segments * sizeof(byte*) );
c	bufferForDebugging = (byte*)calloc( PRINT_BUFFER_SIZE , sizeof(byte) );
c	snippets = (actionHistorySnippet*)calloc( mainArray.upperLimit , sizeof(actionHistorySnippet) );

*/	

/*
	What to change if grammar changes:
	 MAX_ACTIONS 3 	// max num of actions per state per token
	 MAX_RHS_COUNT 3	// max element in array rhscount
	 NRULES 19		// number of grammar rules
	 NSTATES 24		// NSTATES -- Number of States
	 INITIAL_STATE 0   // Usually zero
	 TERMINAL_STATE 5  // the state with Accept
	 NTOKENS  7 	// NTOKENS -- Number of terminals.  
	 NNTERMS  3 		// NNTERMS -- Number of nonterminals.  
	 tokenCode[]	// I don't know where to get this from
	 tokenName[]	// copy from bison's yytname
	 lhs[]			// copy from bison's yyr1
	 rhs[]			// copy from bison's yyrhs
	 prhs[]			// copy from bison's yyprhs
	 rhscount[]		// copy from bison's yyr2
	 table[][][]	// must be auto-generated
	 nra[][]		// must be auto-generated   

for table, if it is generated by parsing y.output, note that a gap will form
in the table if the terminal state is neglected. so all shifts to states 
greater than YYterminalState must decremented by 1

*/

	/*
	// To Print Macro
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


/*---------------------------.
| Grammar-specific Arrays.   |
`---------------------------*/
__constant__ const char tokenCode[] =
	{
       0,  1,   2,    97,   99,   103,   117
	};	

__constant__ char tokenName[NSYMBOLS+1][32] =
	{
	  "$end", "error", "$undefined", "a", "c", "g", "u", "$accept", "S_", "S", 0
	};	

/* lhs[ruleNum+1] -- Number of symbol that rule ruleNum+1 derives.
	+1 because the first element is dummy. this is because bison's
	action table needs 0 to be mean error and -1 to mean the action
	"reduce by rule 0" */
__constant__ const byte lhs[] =
{
       0,     7,     8,     9,     9,     9,     9,     9,     9,     9,
       9,     9,     9,     9,     9,     9,     9,     9,     9,     9
};

/* rhscount[ruleNum+1] -- Number of symbols composing right hand side of rule ruleNum+1.  
	+1 because the first element is dummy for whatever reason bison intended  */
__constant__ const byte rhscount[] =
{
       0,     2,     1,     2,     1,     1,     1,     1,     2,     2,
       2,     2,     2,     2,     3,     3,     3,     3,     3,     3
};

/* PRHS[YYN] -- Index of the first RHS symbol of rule number ruleNum+1 in
   RHS.  */
__constant__ const byte prhs[] =
{
       0,     0,     3,     5,     8,    10,    12,    14,    16,    19,
      22,    25,    28,    31,    34,    38,    42,    46,    50,    54
};

/* RHS -- A `-1'-separated list of the rules' RHS.  */
__constant__ const signed char rhs[] =
{
       8,     0,    -1,     9,    -1,     9,     9,    -1,     3,    -1,
       4,    -1,     5,    -1,     6,    -1,     3,     6,    -1,     6,
       3,    -1,     4,     5,    -1,     5,     4,    -1,     5,     6,
      -1,     6,     5,    -1,     3,     9,     6,    -1,     6,     9,
       3,    -1,     4,     9,     5,    -1,     5,     9,     4,    -1,
       5,     9,     6,    -1,     6,     9,     5,    -1
};

/* DEFACT[stateNum] -- default ruleIndex (ruleNum+1) to reduce with in state
   stateNum when TABLE doesn't specify something else to do.  Zero means the
   default is an error. In parallel, this array will be referenced when a 
   reduce/reduce conflict is detected as the action returned would be the
   second reduce option. */
__constant__ byte defact[] =
{
       0,     4,     5,     6,     7,     0,     2,     8,     0,    10,
       0,    11,    12,     0,     9,    13,     0,     1,     3,    14,
      16,    17,    18,    15,    19
};

/* table[][][] --  symbols in accordance with array tokenName.  
   note that there is one state ( state 3 here ) that is the terminal state.
   this state has only 1 action possible, i.e., accept. so u either imbed 
   that in code or put an accept code in the array. I chose the latter and
   put 127 on state 3
*/

__constant__ byte table[NSTATES][NSYMBOLS+1][MAX_ACTIONS] =
{
    /* Token #          0,             1,             2,             3,             4,             5,             6,             7,             8,             9,            10,  */
//                     eof           undef           err             a              c              g              u             acc             S_             S
  /* state 0 */ { { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 , 0 , 0 }, { 2 , 0 , 0 }, { 3 , 0 , 0 }, { 4 , 0 , 0 }, { 0 , 0 , 0 }, { 5 , 0 , 0 }, { 6 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 1 */ { {-3 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 ,-3 , 0 }, { 2 ,-3 , 0 }, { 3 ,-3 , 0 }, { 7 ,-3 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 8 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 2 */ { {-4 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 ,-4 , 0 }, { 2 ,-4 , 0 }, { 9 ,-4 , 0 }, { 4 ,-4 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {10 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 3 */ { {-5 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 ,-5 , 0 }, {11 ,-5 , 0 }, { 3 ,-5 , 0 }, {12 ,-5 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {13 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 4 */ { {-6 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {14 ,-6 , 0 }, { 2 ,-6 , 0 }, {15 ,-6 , 0 }, { 4 ,-6 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {16 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 5 */ { {127, 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 6 */ { {-1 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 , 0 , 0 }, { 2 , 0 , 0 }, { 3 , 0 , 0 }, { 4 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {17 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 7 */ { {-7 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {14 ,-6 ,-7 }, { 2 ,-6 ,-7 }, {15 ,-6 ,-7 }, { 4 ,-6 ,-7 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {16 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 8 */ { { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 , 0 , 0 }, { 2 , 0 , 0 }, { 3 , 0 , 0 }, {18 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {17 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 9 */ { {-9 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 ,-5 ,-9 }, {11 ,-5 ,-9 }, { 3 ,-5 ,-9 }, {12 ,-5 ,-9 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {13 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 10*/ { { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 , 0 , 0 }, { 2 , 0 , 0 }, {19 , 0 , 0 }, { 4 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {17 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 11*/ { {-10, 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 ,-4 ,-10}, { 2 ,-4 ,-10}, { 9 ,-4 ,-10}, { 4 ,-4 ,-10}, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {10 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 12*/ { {-11, 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {14 ,-6 ,-11}, { 2 ,-6 ,-11}, {15 ,-6 ,-11}, { 4 ,-6 ,-11}, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {16 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 13*/ { { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 , 0 , 0 }, {20 , 0 , 0 }, { 3 , 0 , 0 }, {21 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {17 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 14*/ { {-8 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 ,-3 ,-8 }, { 2 ,-3 ,-8 }, { 3 ,-3 ,-8 }, { 7 ,-3 ,-8 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 8 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 15*/ { {-12, 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 ,-5 ,-12}, {11 ,-5 ,-12}, { 3 ,-5 ,-12}, {12 ,-5 ,-12}, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {13 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 16*/ { { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {22 , 0 , 0 }, { 2 , 0 , 0 }, {23 , 0 , 0 }, { 4 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {17 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state T */  /*blank.. in bison, this is the terminal state but for this application, it is useless .. terminal state here is state 5 */
  /* state 17*/ { {-2 , 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 ,-2 , 0 }, { 2 ,-2 , 0 }, { 3 ,-2 , 0 }, { 4 ,-2 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {17 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 18*/ { {-13, 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {14 ,-6 ,-13}, { 2 ,-6 ,-13}, {15 ,-6 ,-13}, { 4 ,-6 ,-13}, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {16 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 19*/ { {-15, 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 ,-5 ,-15}, {11 ,-5 ,-15}, { 3 ,-5 ,-15}, {12 ,-5 ,-15}, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {13 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 20*/ { {-16, 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 ,-4 ,-16}, { 2 ,-4 ,-16}, { 9 ,-4 ,-16}, { 4 ,-4 ,-16}, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {10 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 21*/ { {-17, 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {14 ,-6 ,-17}, { 2 ,-6 ,-17}, {15 ,-6 ,-17}, { 4 ,-6 ,-17}, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {16 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 22*/ { {-14, 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 ,-3 ,-14}, { 2 ,-3 ,-14}, { 3 ,-3 ,-14}, { 7 ,-3 ,-14}, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 8 , 0 , 0 }, { 0 , 0 , 0 }, },   
  /* state 23*/ { {-18, 0 , 0 }, { 0 , 0 , 0 }, { 0 , 0 , 0 }, { 1 ,-5 ,-18}, {11 ,-5 ,-18}, { 3 ,-5 ,-18}, {12 ,-5 ,-18}, { 0 , 0 , 0 }, { 0 , 0 , 0 }, {13 , 0 , 0 }, { 0 , 0 , 0 }, },   
};

/* nra[stateNum] -- Number of possible Reduce Actions for state StateNum 
    -1 is the terminal state ..
    This can be useful in bigger grammars.. used only for checking */
__constant__ const byte nra[NSTATES][NSYMBOLS+1] = 
{
	/* Token #      0,     1,     2,     3,     4,     5,     6,     7,     8,  */
//                 eof   undef   err     a      c      g      u     acc     S_     S
  /* state 0 */ {   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  , },
  /* state 1 */ {   1  ,   0  ,   0  ,   1  ,   1  ,   1  ,   1  ,   0  ,   0  ,   0  ,   0  , },
  /* state 2 */ {   1  ,   0  ,   0  ,   1  ,   1  ,   1  ,   1  ,   0  ,   0  ,   0  ,   0  , },
  /* state 3 */ {   1  ,   0  ,   0  ,   1  ,   1  ,   1  ,   1  ,   0  ,   0  ,   0  ,   0  , },
  /* state 4 */ {   1  ,   0  ,   0  ,   1  ,   1  ,   1  ,   1  ,   0  ,   0  ,   0  ,   0  , },
  /* state 5 */ {   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  , },
  /* state 6 */ {   1  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  , },
  /* state 7 */ {   1  ,   0  ,   0  ,   2  ,   2  ,   2  ,   2  ,   0  ,   0  ,   0  ,   0  , },
  /* state 8 */ {   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  , },
  /* state 9 */ {   1  ,   0  ,   0  ,   2  ,   2  ,   2  ,   2  ,   0  ,   0  ,   0  ,   0  , },
  /* state 10*/ {   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  , },
  /* state 11*/ {   1  ,   0  ,   0  ,   2  ,   2  ,   2  ,   2  ,   0  ,   0  ,   0  ,   0  , },
  /* state 12*/ {   1  ,   0  ,   0  ,   2  ,   2  ,   2  ,   2  ,   0  ,   0  ,   0  ,   0  , },
  /* state 13*/ {   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  , },
  /* state 14*/ {   1  ,   0  ,   0  ,   2  ,   2  ,   2  ,   2  ,   0  ,   0  ,   0  ,   0  , },
  /* state 15*/ {   1  ,   0  ,   0  ,   2  ,   2  ,   2  ,   2  ,   0  ,   0  ,   0  ,   0  , },
  /* state 16*/ {   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  ,   0  , },
  /* state 17*/ {   1  ,   0  ,   0  ,   1  ,   1  ,   1  ,   1  ,   0  ,   0  ,   0  ,   0  , },
  /* state 18*/ {   1  ,   0  ,   0  ,   2  ,   2  ,   2  ,   2  ,   0  ,   0  ,   0  ,   0  , },
  /* state 19*/ {   1  ,   0  ,   0  ,   2  ,   2  ,   2  ,   2  ,   0  ,   0  ,   0  ,   0  , },
  /* state 20*/ {   1  ,   0  ,   0  ,   2  ,   2  ,   2  ,   2  ,   0  ,   0  ,   0  ,   0  , },
  /* state 21*/ {   1  ,   0  ,   0  ,   2  ,   2  ,   2  ,   2  ,   0  ,   0  ,   0  ,   0  , },
  /* state 22*/ {   1  ,   0  ,   0  ,   2  ,   2  ,   2  ,   2  ,   0  ,   0  ,   0  ,   0  , },
  /* state 23*/ {   1  ,   0  ,   0  ,   2  ,   2  ,   2  ,   2  ,   0  ,   0  ,   0  ,   0  , },
};


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

/*---------------------------------------------------.
|       STACK and Memory Management Functions        |
`---------------------------------------------------*/

__device__ int stackMainLock = 0;

__device__ void superCudaMemcpy( byte *destination , byte *source , unsigned int len)
{
	unsigned int counter;
	
	for(counter = 0; counter < len; counter++)
		destination[counter] = source[counter];
}

#if MY_STACK_MANAGEMENT

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

__device__ void copyElement( sps *spsArrayByte , unsigned int destination , unsigned int source )
{
	byte *spsArraySrc = (byte*)(&(((sps*)spsArrayByte)[source]));
	byte *spsArrayDes = (byte*)(&(((sps*)spsArrayByte)[destination]));
	
	superCudaMemcpy( spsArrayDes , spsArraySrc , sizeof(sps) );
}

__global__ void preprocessSpsArr( sps *spsArray , sps* oldArray , int len , int *sucCount, int *accCount ) 
{
	int current = 0;
	int tail = 0;
	int lCount = 0;
	int lAccCount = 0;
	
	*(sucCount) = 0;
	*(accCount) = 0;
	
		
	for(current = 0; current < len; current++)
	{
		if(spsArray[current].statusFlag == SPS_SUCCESS)
		{
			lCount++;

			if(tail != current)
			{
				copyElement( spsArray , tail , current );
			}

			tail++;
		}
		else if(spsArray[current].statusFlag == SPS_ERROR)
		{
			if( spsArray[current].stackID >= 0 && spsArray[current].stackID <= mainStack.segments)
				deallocateStack( spsArray[current].stackID );
		}
		else if(spsArray[current].statusFlag == SPS_ACCEPT)
		{
			lAccCount++;
			// TODO: copy it to some reservoire or haven, then when there are 0 successful blocks, this reservoire is dumped by main()
			//copyElement( spsArray , tail , current );
			if( spsArray[current].stackID >= 0 && spsArray[current].stackID <= mainStack.segments)
				deallocateStack( spsArray[current].stackID );			
		}
		
	}
	
	*sucCount = lCount;
	*accCount = lAccCount;
	
	for(current = 0; current < len / MAX_ACTIONS; current++)
		deallocateStack( oldArray[current].stackID );
}

#else

__global__ void preprocessSpsArr( sps *spsArray , sps* oldArray , int len , int *sucCount, int *accCount ) 
{
	int current = 0;
	int lCount = 0;
	int lAccCount = 0;
	
	*(sucCount) = 0;
	*(accCount) = 0;
	
		
	for(current = 0; current < len; current++)
	{
		if(spsArray[current].statusFlag == SPS_SUCCESS)
		{
			lCount++;
		}
		else if(spsArray[current].statusFlag == SPS_ACCEPT)
		{
			lAccCount++;
			// TODO: copy it to some reservoire or haven, then when there are 0 successful blocks, this reservoire is dumped by main()
			//copyElement( spsArray , tail , current );
		}
	}
	
	*sucCount = lCount;
	*accCount = lAccCount;
}

__global__ void initializeFirstSPS( sps firstSps, sps *array, byte *p )
{
	cuPrintf( "STACK ID %d State %d\n" , array->stackID , array->stateNum );	
	*array = firstSps;
	array->stackID = p;
	cuPrintf( "STACK ID %d State %d\n" , array->stackID , array->stateNum );	

	array->stateNum = 0;
	array->offset = 0;
	array->stackID[(array->offset)++] = 0;
	cuPrintf( "STACK ID %d State %d\n" , array->stackID , array->stateNum );
}

#endif

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

#if MY_STACK_MANAGEMENT
//STUFF

__global__ void initializeSPSArray( sps init , sps *array )
{
	
	// these three prints are for debugging the initialization of sps blocks
	#if DEV_DEBUG_STACK
	//cuPrintf( "STACK ID %d State %d\n" , array->stackID , array->stateNum );	
	#endif
	
	*array = init;
	array->stackID = allocateStack();

	#if DEV_DEBUG_STACK
	//cuPrintf( "STACK ID %d State %d\n" , array->stackID , array->stateNum );	
	#endif
	
	PUSH( array->stackID , (uint16)(array->stateNum) );
	
	#if DEV_DEBUG_STACK
	//cuPrintf( "STACK ID %d State %d\n" , array->stackID , array->stateNum );
	#endif
}
#endif

/*-------------------------------------.
|     Testing Stack Implementaion      |
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

/*----------------------------.
| Auxiliary Device Functions  |
`----------------------------*/

__global__ void translate( byte *translated, byte *raw )
{
	uint16 code;
	uint16 currChar;
		
	currChar = (uint16) raw[blockIdx.x];
	code = tokenCode[blockIdx.y];

	
	if (currChar == code)
	{

		translated[blockIdx.x] = (byte)blockIdx.y;
		return;
	}
}

__device__ byte negate( byte b )
{
	return ( (b^0xFF)+1 ); 
}

__device__ bool isPositive( byte b )
{
	return ( (b & 0x80) != 0x80 );
}

__device__ bool isNegative( byte b )
{
	return ( b & 0x80 );
}


/*--------------------------.
| Main Multicore Function.  |
`--------------------------*/

__global__ void parse( byte *translatedInputString, sps *inSps, sps *outSps, byte *inStackSpace, byte *outStackSpace, int totalBlocks ) 
{

	// variable definitions
	int i;
	actionHistorySnippet action;
	int inSpsIndex; // inSps[inSpsIndex] is the input sps block for this cudaBlock.
	byte lStateNum; // number of the state on top of the stack
	int childNum; // number of the cudaBlock with respect to cudaBlocks using this inSps block
	uint16 currCharIndex;
	byte tokenIndex;	// index in array "tokenCode" of token at hand.
	byte actionToPerform;	// action this child would perform
	byte actionOfChild0;	// action child number 0 would perform
	int lnra;
	//int lenBuffer;
	//char strBuffer[1024];
	byte gotostate = STATE_INVALID;
	
	#if GRID_PARSE
	int lBlockNum = (blockIdx.y * gridDim.x) + blockIdx.x;
	#else
	int lBlockNum = blockIdx.x;
	#endif

	// kill excess cudaBlocks
	if(lBlockNum >= totalBlocks)
		PARSE_KILLSPS;

	// variable initialization: a precaution since nvcc does work in mysterious ways.
	inSpsIndex = lBlockNum / MAX_ACTIONS;
	lStateNum = inSps[inSpsIndex].stateNum;
	if(lStateNum >= NSTATES)  // just checking 
	{
		#if DEV_DEBUG_EXCEPTION
		cuPrintf("!!! current StateNum %d out of range !!!\n", (byte)lStateNum);
		#endif
		PARSE_KILLSPS;
	}
	currCharIndex = inSps[inSpsIndex].currCharIndex;
	tokenIndex = translatedInputString[currCharIndex];
	lnra = nra[lStateNum][tokenIndex];
	actionOfChild0 = table[lStateNum][tokenIndex][0];
	childNum = lBlockNum % MAX_ACTIONS;
	actionToPerform = table[lStateNum][tokenIndex][childNum];
	
	//We set the status flag of the SPS to be ERROR just in case.
	//We also assign an invalid stack ID to make sure that preprocessSpsArr wouldn't deallocate a valid ID that
	//belong to another SUCCESSFUL SPS.
	#if MY_STACK_MANAGEMENT
	outSps[lBlockNum].stackID = STACK_ID_INVALID;
	#endif
	outSps[lBlockNum].statusFlag = SPS_ERROR;
	
	/* 
		Dealing with parallel processing is tricky when it comes to accessing data.
		That's why, for example, we don't read from the stack of the inSps but 
		copy it to a stack exclusive to outSps then pop and push as we like.
		So you have to make sure that any device or global function does not 
		access, in parallel, a public variable, like the stack buffer, or a local pointer pointing 
		to a public source, like dev_translatedInputString, without using locks. 
	*/

	/* 
	    Another very important bug. signed char, for some very mysterious
	    reason, is read from the array as an unsigned char, i.e., -4 
	    is naturally stored as 252 but when read and assigned to a signed
	    char, it does not revert to -4 ! instead it is printed out as a comma;
	    this happens because we print '0' + 252 which tranlates to 48 + 252 =
	    44, 48 being the ascii code of '0' and 44 that of a comma.
	    Therefore, I will not use any signed types, but I will crudly use it
	    as it is. for example isNegative would check if it is > 127 and 
	    negate would perform a low-level 2's compliment.
	*/

	/* 
	    There is a bug here. The array "table" does not assign zeros 
	    to unassigned values, e.g., in {0, -3 }, since the maximum is 3 numbers,
	    we don't know the third to be surely zero. A solution is to use
	    the array "nra" or "number of reduce actions", to kill, early on,
	    cudaBlocks that we don't need. HOWEVER, expert opinion states that 
	    it is more efficient to just zero unassigned places in "table".
	    For the record, I will implement both solutions just in case the auto-
	    generated table doesn't hold the same format.
	*/

	// Precautionary Checks

	/*
	// if the first action is 0 then all should be killed
	if ( actionOfChild0 == 0 )  
		PARSE_KILLSPS
	
	// if first action is positive, then children > nra should die
	else if( (actionOfChild0 & 0x80) != 0x80 && childNum > lnra )
		PARSE_KILLSPS
		
	// if it is negative, then blocks > nra-1 should die
	else if( childNum > lnra -1 )
		PARSE_KILLSPS
	*/
	
	//if( lBlockNum > 1500 )
		//PARSE_KILLSPS
	
	if( actionToPerform == ACTION_TABLE_ACCEPT &&
		lStateNum != TERMINAL_STATE )
	{
		#if DEV_DEBUG_EXCEPTION
		cuPrintf("!!! Reached Accept in Non-Terminal State# %d !!!\n", lStateNum);
		#endif
		PARSE_KILLSPS;
	} 
	
	// END Precautionary Checks

	if( actionToPerform == ACTION_TABLE_ERROR )
	{// Error
		PARSE_KILLSPS
	}
	else if( actionToPerform == ACTION_TABLE_ACCEPT ) 
	{// Accept

		// Prepare this cudaBlock's output which is an sps struct called outSps
		outSps[lBlockNum].statusFlag = SPS_ACCEPT;
		outSps[lBlockNum].currCharIndex = currCharIndex;
		
		//// Record Action in the History of Time
		//action.blockID = (byte) lBlockNum;
		//action.charIndex = currCharIndex;
		//action.actionTaken = actionToPerform;
		//addActionTaken(action);		
		
		// Debug Clean Print: Accept
		#if DEV_DEBUG_CLEAN
		cuPrintf("T[%d]=%c: Accept\n", currCharIndex, tokenName[tokenIndex][0]);
		#endif
		// Debug Crude Print: Accept
		#if DEV_DEBUG_CRUDE
		cuPrintf("\tT[%d]=%c\tStt=%2d\tAct=Acc\t                    \t|\n", currCharIndex, tokenName[tokenIndex][0], lStateNum );
		#endif
	}
	else if ( isPositive(actionToPerform) && actionToPerform < NSTATES )	// if positive and in range
	{// Shift
		gotostate = actionToPerform; // the state to go to with shift
		
		// Prepare this cudaBlock's output which is an sps struct called outSps
		outSps[lBlockNum].statusFlag = SPS_SUCCESS;
		outSps[lBlockNum].currCharIndex = currCharIndex + 1;
		outSps[lBlockNum].stateNum = gotostate;
		
		// Allocate stack for outSps
		#if MY_STACK_MANAGEMENT
		    outSps[lBlockNum].stackID = allocateStack();
		if( outSps[lBlockNum].stackID == STACK_ID_INVALID)
		{
			#if DEV_DEBUG_EXCEPTION
			cuPrintf("!!! ERROR allocating stack !!!\n");
			#endif
			PARSE_KILLSPS;
		}
		#else 
		outSps[lBlockNum].stackID = outStackSpace + lBlockNum;
		#endif
		
		// copy stack from inSps (Parent) to outSps (Child)
		// and report if error
		#if MY_STACK_MANAGEMENT
		#if DEV_DEBUG_STACK
		cuPrintf("COPY Stk#%d to Stk#%d\n" , inSps[inSpsIndex].stackID , outSps[lBlockNum].stackID );
		#endif
		if ( !copyStack( outSps[lBlockNum].stackID , inSps[inSpsIndex].stackID ))
		{
			#if DEV_DEBUG_EXCEPTION
			cuPrintf("!!! ERROR copyStack !!!\n");
			#endif
			PARSE_KILLSPS;
		}
		#else
		superCudaMemcpy(outSps[lBlockNum].stackID, inSps[inSpsIndex].stackID, STACK_SEGMENT_SIZE);
		#endif

		// Push token into stack together with state number. Push both bytes as one uint16: Hi is tokenIndex, Lo is stateNum
		#if MY_STACK_MANAGEMENT		
		if( !PUSH( outSps[lBlockNum].stackID , (uint16) ((tokenIndex << 8) | gotostate )) )
		{
			#if DEV_DEBUG_EXCEPTION
			cuPrintf("Error PUSHING %4x in ID %d !!\n" , (uint16) ((tokenIndex << 8) | gotostate) , outSps[lBlockNum].stackID );
			#endif
			PARSE_KILLSPS;
		}
		#else
		*(outSps[lBlockNum].stackID + (outSps[lBlockNum].offset++)) = gotostate;
		#endif
		
		// Report PUSH 
		#if DEV_DEBUG_STACK && MY_STACK_MANAGEMENT
		cuPrintf( "PUSH %4x in Stk#%d\n" , (uint16) ((tokenIndex << 8) | gotostate ) , outSps[lBlockNum].stackID );
		#endif
		// Record Action in the History of Time
		//action.blockID = (byte) lBlockNum;
		//action.charIndex = currCharIndex;
		//action.actionTaken = actionToPerform;
		//addActionTaken(action);
		
		// Debug Clean Print: Shift
		#if DEV_DEBUG_CLEAN
		cuPrintf("T[%d]=%c: Shift and goto State # %d\n", currCharIndex, tokenName[tokenIndex][0], actionToPerform);
		#endif

		// Debug Crude Print: Shift
		#if DEV_DEBUG_CRUDE
		cuPrintf("\tT[%d]=%c\tStt=%2d\tAct=S%d\t              \tNxt=%2d\t|\n", currCharIndex, tokenName[tokenIndex][0], lStateNum, actionToPerform, gotostate);
		#endif
		
	}
	else if( isNegative(actionToPerform) && negate(actionToPerform) < NRULES )	// if negative and in range
	{// Reduce
	
		unsigned short topStateBeforePop; // number of the state on top of the stack before any pop of a reduce
		unsigned short topStateAfterPop; // number of the state on top of the stack after all pops of a reduce
		unsigned short tempPOP;
		
		/* ruleNum: Number of the rule to reduce by.
		   never use ruleNum directly as an index to arrays.
		   instead, use ruleIndex.
		*/
		byte ruleNum; // number of the rule to reduce by
		byte ruleIndex; // ruleNum + 1
		byte nterm;   // nonterminal to reduce to.

		ruleNum = negate(actionToPerform); 
		ruleIndex = ruleNum + 1;		
		nterm = lhs[ruleIndex];  // nonterminal to reduce to.

		// Prepare this cudaBlock's output which is an sps struct called outSps
		outSps[lBlockNum].statusFlag = SPS_SUCCESS;
		outSps[lBlockNum].currCharIndex = currCharIndex;

		// Allocate stack for outSps
		#if MY_STACK_MANAGEMENT
		    outSps[lBlockNum].stackID = allocateStack();
		if( outSps[lBlockNum].stackID == STACK_ID_INVALID)
		{
			#if DEV_DEBUG_EXCEPTION
			cuPrintf("ERROR allocating stack !!\n");
			#endif
			PARSE_KILLSPS;
		}
		#else
		outSps[lBlockNum].stackID = outStackSpace + lBlockNum;		
		#endif
		
		// copy stack from inSps (Parent) to outSps (Child)
		#if MY_STACK_MANAGEMENT
		#if DEV_DEBUG_STACK
		cuPrintf("COPY Stk#%d to Stk#%d\n" , inSps[inSpsIndex].stackID , outSps[lBlockNum].stackID );
		#endif		
		if( !(copyStack( outSps[lBlockNum].stackID , inSps[inSpsIndex].stackID )) )
		{
			#if DEV_DEBUG_EXCEPTION
			cuPrintf("!!! ERROR copyStack !!!, blocnum %d outof %d\n", lBlockNum, totalBlocks);
			#endif
			PARSE_KILLSPS;
		}
				
		// test and kill if out of range
		topStateBeforePop = PEEK (outSps[lBlockNum].stackID);
		if(topStateBeforePop < 0 || topStateBeforePop >= NSTATES)
		{
			#if DEV_DEBUG_EXCEPTION
			cuPrintf("!!! Bfr State out of range !!!\n");
			#endif
			PARSE_KILLSPS;
		}
		#else
		superCudaMemcpy(outSps[lBlockNum].stackID, inSps[inSpsIndex].stackID, STACK_SEGMENT_SIZE);
		#endif

		// Pop tokens
		for (i=0; i < rhscount[ruleIndex]; i++)
		{
			#if MY_STACK_MANAGEMENT
			tempPOP = POPshort (outSps[lBlockNum].stackID);
			#if DEV_DEBUG_STACK
			cuPrintf("POP#%d=%4x from Stk#%d\n" , i + 1 , tempPOP , outSps[lBlockNum].stackID );
			#endif
			
			// Check from Rules' RHS if popped tokens are as expected
			// and report error if not
			if( tempPOP >> 8 != rhs[prhs[ruleIndex]+rhscount[ruleIndex]-1-i] )
			{
				#if DEV_DEBUG_EXCEPTION
				cuPrintf("!!! Popped token %d. Expected %d !!!\n", tempPOP >> 8, rhs[prhs[ruleIndex]+rhscount[ruleIndex]-1-i]);
				#endif
				PARSE_KILLSPS;
			}
			else if ( 255 == rhs[prhs[ruleIndex]+i] )
			{
				#if DEV_DEBUG_EXCEPTION
				cuPrintf("!!! Read RHS (-1) !!!\n");
				#endif
				PARSE_KILLSPS;
			}
			#else
			*(outSps[lBlockNum].stackID + (--(outSps[lBlockNum].offset))) = 0;
			#endif
		}
		// Here i peek only with a byte size because I only care about the lower byte which is stateNum
		#if MY_STACK_MANAGEMENT
		topStateAfterPop = PEEK (outSps[lBlockNum].stackID);
		if(topStateAfterPop < 0 || topStateAfterPop >= NSTATES)
		{
			#if DEV_DEBUG_EXCEPTION
			cuPrintf("!!! Aft State out of range !!!\n");
			#endif
			PARSE_KILLSPS;
		}
		#else
		topStateAfterPop = *(outSps[lBlockNum].stackID + outSps[lBlockNum].offset -1);
		#endif

		// get the goto-state from the table
		gotostate = table[topStateAfterPop][nterm][0];
		if(gotostate < 0 || gotostate >= NSTATES)
		{
			#if DEV_DEBUG_EXCEPTION
			cuPrintf("!!! Goto State out of range !!!\n");
			#endif
			PARSE_KILLSPS;
		}
		
		// Push nonterminal into stack: Hi is Symbol index, Lo is stateNum
		#if MY_STACK_MANAGEMENT
		if( !PUSH( outSps[lBlockNum].stackID , (uint16)((nterm << 8) | gotostate) ))
		{
			#if DEV_DEBUG_EXCEPTION
			cuPrintf("!!! Error PUSHING %4x in ID %d !!!\n" , (uint16)((nterm << 8) | gotostate) , outSps[lBlockNum].stackID );
			#endif
			PARSE_KILLSPS;
		}
		#if DEV_DEBUG_STACK
		cuPrintf("PUSH %4x in Stk#%d\n" , (uint16)((nterm << 8) | gotostate) , outSps[lBlockNum].stackID );
		#endif
		#else
		*(outSps[lBlockNum].stackID + (outSps[lBlockNum].offset++)) = gotostate;
		#endif
		
		// set child's stateNum as the goto-state
		outSps[lBlockNum].stateNum = gotostate;
		
		// Record 2 Actions in the History of Time: 1-reduce, 2-shift nterm
		  // Record Action #1: Reduced by Rule x
		//action.blockID = (byte) lBlockNum;
		//action.charIndex = currCharIndex;
		//action.actionTaken = actionToPerform;
		//addActionTaken(action);
		
		/*
		  // Record Action #2: Shifted Non-Terminal X
		actionHistorySnippet action2;
		action2.blockID = (byte) lBlockNum;
		action2.charIndex = currCharIndex;
		action2.actionTaken = gotostate;
		addActionTaken(action2);
		*/
		
		// Debug Clean Print: Reduce
		#if DEV_DEBUG_CLEAN
		cuPrintf("T[%d]=%c: Reduce by Rule # %d\n", currCharIndex, tokenName[tokenIndex][0], (byte)ruleNum);
		#endif
		
		// Debug Crude Print: Reduce
		#if DEV_DEBUG_CRUDE
		cuPrintf("\tT[%d]=%c\tStt=%2d\tAct=R%d\t#Pops=%d\taft=%2d\tNxt=%2d\t|\n", currCharIndex, tokenName[tokenIndex][0], lStateNum, ruleNum, i, topStateAfterPop, gotostate);
		#endif
	}
	else
	{
		#if DEV_DEBUG_EXCEPTION
		if(actionToPerform & 0x80)
			 cuPrintf("!!! Action -%d out of range !!!\n", negate(actionToPerform));
		else cuPrintf("!!! Action %d out of range !!!\n", actionToPerform);
		#endif
		PARSE_KILLSPS;	
	}
}



/*---------------------------.
| Auxiliary Host Functions.  |
`---------------------------*/

static int HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
		if ( !errorCaught )
		{
			// print Error message and other info helping debug
			printf( "\n%s in %s at line %d. Iteration# %d\n", cudaGetErrorString( err ), file, line, mainIteration );
		}
		//exit( EXIT_FAILURE );
		errorCaught = true;
		return 1;
		
    }
    return 0;
}

void freeAll()
{
}

int getFileSize (FILE *fp)
{
	int fileSize = -1;
	if (fp)
	{
		fseek(fp, 0L, SEEK_END); //seek to end of file
		fileSize = ftell(fp);   //get size of file
		fseek(fp, 0L, SEEK_SET); //reseek to beginning of file
	}
	return fileSize;	
}

/*--------------------------------------------------------------.
| BGN: main() 													|
|                                                               |
`--------------------------------------------------------------*/

int main(void) {

/*----------------------.
| Variable Defintions.  |
`----------------------*/
	int success=0;
	int fileSize=0;
	char *youts=NULL;
	char *inputString=NULL;
	char *translatedInputString=NULL;
	char currChar;
	sps initialSPS;
	sps *dev_spsArrOut;
	sps *dev_spsArrIn;
	byte *dev_inSpsStackSpace;	// pointer for stacks
	byte *dev_outSpsStackSpace;
	int *dev_successCount;
	int *dev_acceptCount;
	uint16 *dev_tokenIndex;
	int lastSuccessCount = 1;   // for the first iteration of the main loop to run correctly
	int lastAcceptCount = 0;
	int lastSegmentsNeeded = 0;
	int totalAcceptCount = 0;
	int bottleNeckIteration;
	int N=1;	// number of parallel instances to call
	int i=0;
	byte *bufferForDebugging;
	uint16 currCharIndex;
	float elapsedTime;
	float totalExecTime = 0;
	float kernel1Time = 0;
	float kernel2Time = 0;
	
/*-------------------------------------.
| Init events to measure running time  |
`-------------------------------------*/
	
	cudaEvent_t start, stop, kernel1Timer, kernel2Timer;
	HANDLE_ERROR( cudaEventCreate( &start ) );
	HANDLE_ERROR( cudaEventCreate( &stop ) );
	HANDLE_ERROR( cudaEventCreate( &kernel1Timer ) );
	HANDLE_ERROR( cudaEventCreate( &kernel2Timer ) );
	HANDLE_ERROR( cudaEventRecord( start, 0 ) );


/*--------------------------.
| Preparing Actions Array.  |
`--------------------------*/
/*
	// Prepare action arrays from bison-generated state space "y.output"
	FILE *youtf = fopen("y.output","rb");
	if(youtf) 
	{
		fileSize = getFileSize(youtf);   //get size of file
		//youts = (char *)malloc (fileSize);	 //allocate memory for input string
		HANDLE_ERROR( cudaHostAlloc( (void**)&youts, fileSize, cudaHostAllocDefault ) );

		success= fscanf(youtf, "%[^<EOF>]", youts);
		printf("\n\n========================================================================+\n");
		if(success) 
		{
			printf("\n\tYacc output read successfully\
				\n\tfile size = %d bytes\
				\n\tscanned string length = %d\n" \
				, fileSize,strlen(youts));
		}
		else 		printf("\n\tYacc output read failed\n");
		printf("\n========================================================================+\n");
	}
	else 
	{
		printf("\n\tFile \"y.output\" Not Found!\n\n");
		exit(0);
	}
	fclose(youtf);

	//tempStr = strstr(youts, "state 8");  
	//printf("\nFOUND:\t%s\n",tempStr);
	


	fileSize=0;
	success=0;
*/

/*-------------------------.
| Preparing Output Stream. |
`-------------------------*/
#if PRINT_TO_FILE
	// Prepare output stream
	FILE *outfile = fopen("out","wb");
	if(outfile) 
	{
		printf("\n\tAll output is being sent to file \"out\"");
		printf("\n========================================================================+\n");
	}
	else 
	{
		printf("\n\tCould Not Create Output File!\n\n");
		exit(0);
	}
#endif

/*-------------------------.
| Preparing Input String.  |
`-------------------------*/

	// Prepare input stream and load it onto device
	FILE *infile = fopen("in","rb");
	if(infile) 
	{
		fileSize = getFileSize(infile);   //get size of file
		//inputString = (char *)malloc (fileSize);	 //allocate memory for input string
		HANDLE_ERROR( cudaHostAlloc( (void**)&inputString, fileSize, cudaHostAllocDefault ) );
		success= fscanf(infile, "%[acgu]", inputString);
		inputSize = strlen(inputString);
		if(PRINT_TO_FILE) printf("\n\tInput File Read Successfully\
				\n\tInput File Size    = %d bytes (including EOF and \\n)\
				\n\tInput String Length= %d characters\n"				 \
				, fileSize,inputSize);
		fprintf(OUTPUT_STREAM, "\n\tInput File Read Successfully\
				\n\tInput File Size    = %d bytes (including EOF and \\n)\
				\n\tInput String Length= %d characters\n"				 \
				, fileSize,inputSize);
		printf("\n========================================================================+\n");
		fprintf(OUTPUT_STREAM, "Input: %s\n", inputString);
	}
	else 
	{
		printf("\n\tInput File Not Found!\n\n");
		exit(0);
	}
	fclose(infile);

/*-------------------------------------------------------------.
| Initialize Stack, Actions History Array, and Everything Else |
`-------------------------------------------------------------*/

	byte *dev_stacktesting;
	byte *testingStack;
	int *spData;
	int spi = 0;

	//spData = (int*)calloc( N_STACK_SEGMENTS , sizeof(byte*) );
	//testingStack = (byte*)calloc( N_STACK_SEGMENTS , sizeof(byte) );
	
	#if MY_STACK_MANAGEMENT
	//Initializing Stack
	initializeStack( STACK_SEGMENT_SIZE , N_STACK_SEGMENTS );
	#endif
	
	initializeActionsTakenArray( (inputSize * 2) + 2 );
	
	initializeDebugging( PRINT_BUFFER_SIZE );
	bufferForDebugging = (byte*)calloc( PRINT_BUFFER_SIZE, sizeof(byte) );
	//HANDLE_ERROR( cudaHostAlloc( (void**)&bufferForDebugging, PRINT_BUFFER_SIZE*sizeof(byte), cudaHostAllocDefault ) );

	//copy input string to device
	cudaMalloc( (void**)&dev_inputString, inputSize+1 );
	cudaMemcpy( dev_inputString, inputString, inputSize+1, cudaMemcpyHostToDevice ); // +1 for the null character
	
/*-------------------------.
| Translate Input String.  |
`-------------------------*/

	// translate tokens into indeces corresponding to arrays "tokenCode" and "tokenName"
	HANDLE_ERROR( cudaMalloc( (void**)&dev_translatedInputString, inputSize+1 ) );
	dim3 grid(inputSize+1,NTOKENS);
	translate<<<grid,1>>>( dev_translatedInputString , dev_inputString );
	HANDLE_ERROR( cudaThreadSynchronize() );

	//copy translated input string to host just for debugging
	//HANDLE_ERROR( cudaMemcpy( &translatedInputString, dev_translatedInputString, inputSize+1, cudaMemcpyDeviceToHost ) ); // +1 for the null character

/*------------.
| Stack Test. |
`------------*/
/*
	byte *dev_stacktesting;
	byte *testingStack;
	int *spData;
	int spi = 0;
	
	spData = (int*)calloc( 100 , sizeof(byte*) );
	spData[0] = 123;
	testingStack = (byte*)calloc( 100 , sizeof(byte) );
	testingStack[0] = 'a';
	
	//Testing Stack Implementaion
	initializeStack( 1024 , 100 );
	cudaMalloc( (void**)&dev_stacktesting, 100 );
	myStackTesting<<<1,1>>>( dev_stacktesting );
	cudaThreadSynchronize();
	cudaMemcpy( testingStack , dev_stacktesting , 100 , cudaMemcpyDeviceToHost ); // +1 for the null character
	cudaMemcpy( spData , mStack.sp , 100 * sizeof(byte*) , cudaMemcpyDeviceToHost ); // +1 for the null character
	HANDLE_FREE( cudaFree, dev_stacktesting );
	printf( "THE RESULT OF THE STACK TESTING IS \"%s\"\n\nSTACK POINTER = %x\n\n", testingStack , mStack.buffer );
	
	for(spi = 0; spi < 1; spi++)
		printf("sp[%d] = %x\n", spi , spData[spi] );
	
	free( spData );
	free( testingStack );
	
	deinitializeStack();
*/

/*------------.
| Main Loop.  |
`------------*/
	
	//Making initially-needed memory in the GPU.
	cudaMalloc( (void**)&dev_successCount , sizeof(int) );
	cudaMalloc( (void**)&dev_acceptCount , sizeof(int) );
	cudaMalloc( (void**)&dev_spsArrIn , sizeof(sps) );
	cudaMalloc( (void**)&dev_inSpsStackSpace, sizeof(byte) * 2);
	
	// Setting up initial SPS
	initialSPS.statusFlag = SPS_SUCCESS;
	initialSPS.currCharIndex = 0;
	initialSPS.stateNum = INITIAL_STATE;
	#if MY_STACK_MANAGEMENT
	initialSPS.stackID = STACK_FIRST_SEGMENT_ID;
	#else
	//initialSPS.stackID = dev_inSpsStackSpace;
	cudaPrintfInit();
	initializeFirstSPS<<<1,1>>>(initialSPS, dev_spsArrIn, dev_inSpsStackSpace);
	HANDLE_ERROR( cudaThreadSynchronize() );
	cudaPrintfDisplay(OUTPUT_STREAM, true);
	cudaPrintfEnd();
	#endif
	
	#if MY_STACK_MANAGEMENT
	cudaPrintfInit();
	initializeSPSArray<<<1,1>>>( initialSPS , dev_spsArrIn );
	HANDLE_ERROR( cudaThreadSynchronize() );
	cudaPrintfDisplay(OUTPUT_STREAM, true);
	cudaPrintfEnd();
	#endif
	
	/* 
		safetyValve is calculated to precisely be the times the loop will
		run given the worst case input which yields worst case Actions:
		Worst case is S R S R S R (n times).... R R R R R (nlog(n) times).
		The +1 in the end is for not reaching zero, so that in the worst 
		case, the loop will exit with safetyValve = 1, which will not 
		run the "if" just after the loop.
	*/
	int safetyValve = 2*inputSize * ( 1 + log2(2.0*inputSize) ) + 1 ;
	
	while( (lastSuccessCount > 0) && (safetyValve > 0) )
	{
		N = lastSuccessCount * MAX_ACTIONS;
		kernel1Time = kernel2Time = 0;
		
		// Allocate sps out
		HANDLE_ERROR( cudaMalloc( (void**)&dev_spsArrOut, sizeof(sps) * N ) );
		
		// initiate device printer
		#if DEV_DEBUG_ANY 
		cudaPrintfInit();  
		#endif

		// intiate timer1 just before device starts execution
		HANDLE_ERROR( cudaEventRecord( kernel1Timer, 0 ) );

		#if MY_STACK_MANAGEMENT
		// CudaMalloc stack space for outSps's stack space
		HANDLE_ERROR( cudaMalloc( (void**)&dev_outSpsStackSpace, sizeof(byte) * N * STACK_SEGMENT_SIZE ));
		#endif
		
		// call cuda
		#if GRID_PARSE
		int tempDim = (int) sqrt(N) +1; // ceiling
		dim3 grid2(tempDim, tempDim);
		parse<<<grid2,1>>>(dev_translatedInputString, dev_spsArrIn, dev_spsArrOut, dev_inSpsStackSpace, dev_outSpsStackSpace, N);
		#else
		parse<<<N,1>>>(dev_translatedInputString, dev_spsArrIn, dev_spsArrOut, dev_inSpsStackSpace, dev_outSpsStackSpace, N);
		#endif		
		if( HANDLE_ERROR( cudaThreadSynchronize() ) == 1)
			safetyValve = SAFETY_PARSE_ERROR;

		// intiate timer2 just before device starts execution and get timer2 - timer1
		HANDLE_ERROR( cudaEventRecord( kernel2Timer, 0 ) );
		HANDLE_ERROR( cudaEventSynchronize( kernel2Timer ) );
		HANDLE_ERROR( cudaEventElapsedTime( &kernel1Time, kernel1Timer, kernel2Timer ) );

		//   post-process the output of this iteration,
		// or pre-process the input  of next iteration, put it however you like.
		// TODO: Parallelize this
		preprocessSpsArr<<<1,1>>>( dev_spsArrOut , dev_spsArrIn , N , dev_successCount, dev_acceptCount );
		if( HANDLE_ERROR( cudaThreadSynchronize() ) == 1)
			safetyValve = SAFETY_PREPROCESS_ERROR;
		
		// stop timer 2 and get time
		HANDLE_ERROR( cudaEventRecord( kernel1Timer, 0 ) );
		HANDLE_ERROR( cudaEventSynchronize( kernel1Timer ) );
		HANDLE_ERROR( cudaEventElapsedTime( &kernel2Time, kernel2Timer, kernel1Timer ) );

		// update total execution time
		totalExecTime += kernel1Time + kernel2Time;
		
		// get the number of successful and accepted blocks of this iteration
		// These are the only memcpys in the loop and they do only 2 ints
		lastSegmentsNeeded = lastSuccessCount; lastSuccessCount = lastAcceptCount = 0;
		cudaMemcpy( &lastSuccessCount , dev_successCount , sizeof(int) , cudaMemcpyDeviceToHost );
		cudaMemcpy( &lastAcceptCount , dev_acceptCount , sizeof(int) , cudaMemcpyDeviceToHost );
		totalAcceptCount += lastAcceptCount;
		lastSegmentsNeeded += lastSuccessCount; // = num of inSps blocks + num of successful outSps blocks
		
		// update maxSuccessCount and maxSegmentsNeeded
		maxSuccessCount = (maxSuccessCount > lastSuccessCount)? maxSuccessCount:lastSuccessCount ;
		if( lastSegmentsNeeded > maxSegmentsNeeded )
		{
			maxSegmentsNeeded = lastSegmentsNeeded;
			bottleNeckIteration = mainIteration;
		}
		// debug print: Iteration Summary
		fprintf(OUTPUT_STREAM, "------------------------------------------------------------------------+\n");
		fprintf(OUTPUT_STREAM, "Iteration # %5d\t%7d Threads   \tExec Time           \t\t\t|\n", mainIteration, N );
		fprintf(OUTPUT_STREAM, "                 \t%7d Successful\tParse     = %6.2f ms\t\t\t|\n", lastSuccessCount, kernel1Time);
		fprintf(OUTPUT_STREAM, "                 \t%7d Accepted  \tPreProcess= %6.2f ms\t\t\t|\n", lastAcceptCount, kernel2Time);
		fprintf(OUTPUT_STREAM, "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t|\n");
		fprintf(OUTPUT_STREAM, "Total Device Exec Time = %6.2f ms\t\t\t\t\t\t\t\t\t\t|\n", totalExecTime);
		fprintf(OUTPUT_STREAM, "------------------------------------------------------------------------+\n");
		#if PRINT_TO_FILE
		printf("------------------------------------------------------------------------+\n");
		printf("Iteration # %d\t%7d Threads\t\tExec Time\t\t\t|\n", mainIteration, N );
		printf("\t\t%7d Successful\tParse= %6.2f ms\t\t|\n", lastSuccessCount, kernel1Time);
		printf("\t\t%7d Accepted\tPreProcess= %6.2f ms\t\t|\n", lastAcceptCount, kernel2Time);
		printf("\t\t\t\t\t\t\t\t\t|\n");
		printf("Total Device Exec Time = %6.2f ms\t\t\t\t\t|\n", totalExecTime);
		printf("------------------------------------------------------------------------+\n");
		#endif
		#if DEV_DEBUG_ANY 
		cudaPrintfDisplay(OUTPUT_STREAM, true);
		cudaPrintfEnd(); 
		#endif
		//fprintf(OUTPUT_STREAM, "\n\t\t\t\t\t\t|\n");
		//time();

		// make this iteration's output next iteration's input.
		HANDLE_FREE( cudaFree, dev_spsArrIn );
		HANDLE_FREE( cudaFree, dev_inSpsStackSpace );
		dev_spsArrIn = dev_spsArrOut;
		dev_inSpsStackSpace = dev_outSpsStackSpace;
		
		mainIteration++;
		safetyValve--;
	}
	
	// say if exited by saftey valve
	printf("========================================================================+\n");
	if(safetyValve==0)printf("Reached Safety Valve.. Exiting. The parse did not complete.\nIncrease the initial value of \"safetyValve\"\n\n");
	
	// print Stack Summary + Bottle Neck Iteration
	//if(!errorCaught)
		if(maxSegmentsNeeded != N_STACK_SEGMENTS)
		{
			fprintf(OUTPUT_STREAM, "  Maximum number of Stack Segments ever needed\
			\n  was %d at iteration# %d while %d were reserved.\n", \
			maxSegmentsNeeded, bottleNeckIteration, N_STACK_SEGMENTS);
			if(PRINT_TO_FILE)
				printf("  Maximum number of Stack Segments ever needed\
				\n  was %d at iteration# %d while %d were reserved.\n", \
				maxSegmentsNeeded, bottleNeckIteration, N_STACK_SEGMENTS);
		}
		else
			printf("ERROR ALLOCATING AT LEAST ONE STACK SEGMENT.\
			\n    The number of stack segments needed in the failed iteration is at most %d.\
			\n    BUT ONLY %d SEGMENTS WERE RESERVED.\n"\
			, maxSuccessCount*MAX_ACTIONS, N_STACK_SEGMENTS);
	
		
	// Copy stack to host
	closeDebugging();
	//cudaMemcpy( bufferForDebugging , printingBuffer , PRINT_BUFFER_SIZE , cudaMemcpyDeviceToHost );
	//printf("\n========================================|");
	//printf("\n++++++Flushing Device Print Buffer++++++|");
	//printf("\n========================================/\n\n%s", bufferForDebugging);
	
	// Copy Parse Trees to host
	//cudaMemcpy(&mainArray , lActionsTaken , sizeof(dynamicArray), cudaMemcpyDeviceToHost );
	//actionHistorySnippet *snippets;
	//snippets = (actionHistorySnippet*)calloc( mainArray.upperLimit , sizeof(actionHistorySnippet) );
	//HANDLE_ERROR( cudaHostAlloc( (void**)&snippets, mainArray.upperLimit*sizeof(actionHistorySnippet), cudaHostAllocDefault ) );
	//cudaMemcpy(snippets , mainArray.buffer , mainArray.upperLimit * sizeof(actionHistorySnippet), cudaMemcpyDeviceToHost );

/*
	//printf("\n=========================================|");
	//printf("\n+++++++++Dumping History of Time+++++++++|");
	//printf("\n=========================================/");
	printf("\n+++++++++  Number of items %d +++++++++|\n\n", mainArray.upperLimit);	
	

	for(i = 0; i < mainArray.index; i+=4)
	{
		if(snippets[i].actionTaken) 
			printf("Character Index for (%d) out of (%d) = %d\n", \
			  i , mainArray.index , snippets[i].actionTaken);
	}
*/	
	/* pseudo code:
		make array of arrays
		traverse mainArray.buffer
			downcast to actionHistorySnippet
			append element to array arrayofarrays[snippet.blockID]
		traverse again
			calculate probability from probability table ( table available for download from my dreams )
			print all. ( not in my dreams )
			go to sleep.	
	*/
	
	
	
	
	//cudaMemcpy( testingStack , dev_stacktesting , N_STACK_SEGMENTS , cudaMemcpyDeviceToHost ); // +1 for the null character
	//cudaMemcpy( spData , mStack.sp , N_STACK_SEGMENTS * sizeof(byte*) , cudaMemcpyDeviceToHost ); // +1 for the null character

	/* Print Stack
	printf( "THE RESULT OF THE STACK TESTING IS \"%s\"\n\nSTACK POINTER = %x\n\n", testingStack , mStack.buffer );	
	for(spi = 0; spi < 1; spi++)
		printf("sp[%d] = %x\n", spi , spData[spi] );
	*/

	// Report output
	// get accepted sps's AND actions dynamicArray and make a big party here.
	if(PRINT_TO_FILE) 
	{
		printf("\n\n");
		printf("  Input of size %d : \"%s\"\n", inputSize, inputString);
		printf("  Number of Parse Trees Accepted = %d\n", totalAcceptCount);
	}
	fprintf(OUTPUT_STREAM, "\n\n");
	fprintf(OUTPUT_STREAM, "  Input of length %d : \"%s\"\n", inputSize, inputString);
	fprintf(OUTPUT_STREAM, "  Number of Parse Trees Accepted = %d\n", totalAcceptCount);

	// get stop time, and display the timing results
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
	start, stop ) );
	if(PRINT_TO_FILE) 
	{
		printf( "  GPU   Running Time: %6.2f ms\n", totalExecTime);
		printf( "  CPU   Running Time: %6.2f ms\n", elapsedTime-totalExecTime);
		printf( "  Total Running Time: %6.2f ms\n", elapsedTime );
	}
	fprintf( OUTPUT_STREAM, "  GPU   Running Time: %6.2f ms\n", totalExecTime);
	fprintf( OUTPUT_STREAM, "  CPU   Running Time: %6.2f ms\n", elapsedTime-totalExecTime);
	fprintf( OUTPUT_STREAM, "  Total Running Time: %6.2f ms\n", elapsedTime );

	#if MY_STACK_MANAGEMENT
	// DeInitialize Stack
	deinitializeStack();
	#endif

	// Free ALL
	HANDLE_FREE( cudaFree, dev_spsArrOut );
	HANDLE_FREE( cudaFree, dev_translatedInputString );
	HANDLE_FREE( cudaFree, printingBuffer );
	HANDLE_FREE( cudaFree, mainArray.buffer );
	HANDLE_FREE( cudaFree, lActionsTaken );
	HANDLE_FREE( cudaFree, dev_inputString );
	HANDLE_FREE( cudaFree, dev_successCount );
	HANDLE_FREE( cudaFree, dev_acceptCount );
	//HANDLE_FREE( cudaFreeHost, bufferForDebugging );
	HANDLE_FREE( free, bufferForDebugging );
	//HANDLE_FREE( cudaFreeHost, snippets );
	//HANDLE_FREE( free, snippets );
	HANDLE_FREE( cudaFreeHost, inputString );
	//HANDLE_FREE( cudaFreeHost, youts );
	//HANDLE_FREE( free, translatedInputString );
	//HANDLE_FREE( free, testingStack );

	//fclose(outfile);
	return 0;
}

/*------------------------------------------------------------------.
| END: main() 														|
|                                                                   |
`------------------------------------------------------------------*/


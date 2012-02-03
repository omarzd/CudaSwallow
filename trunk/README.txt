About this Project:
This is a very simple implementation of a GLR-like parser that works on a 
specific type of input using a specific grammar. I say "GLR-like" because I 
still haven't found a suitable name for it. It is not quite GLR. GLR divides 
upon a conflict but fails when facing a new conflict before resolving the old 
one. This parser does not give up--it goes as deep as possible. It does not 
settle for an approximate solution either. If it is faced with a series of 
conflicts, it keeps dividing and dividing indefinitely; and in the end, it 
outputs all possible parse trees. So if you have a name in mind for such 
parser, please send me your suggestion.

A more detailed report can be found at 
docs.google.com/document/pub?id=1iHpUBOPMwvJrWQHLHhUPHmU2t5KVieq25KItU-UbMpM


--------------------------------------------------------------------------------

Requirements:
- Any Linux distro with kernel 2.5 or higher.
- A Nvidia Graphics Chipset with at least 16 CUDA Cores and 256 MB of memory.
- Nvidia Developer Driver 275 or higher.
- Cuda Toolkit 3.2 or higher.
- cuPrintf Library 2 files: "cuPrintf.cuh" and "cuPrintf.cu"

--------------------------------------------------------------------------------

Compilation:

compile the kernel with nvcc using the following switches:
-w : to supress naive warnings
-lcudart : to link with the famous library
-arch sm_11 : necessary for atomic locks
Example: 
<code> nvcc kernel.cu -w -lcudart -arch sm_11 </code>

There are a number of directives that you can play with before compiling.
They are located at the beginning of the basic.cu file:

- PRINT_TO_FILE -- self explanatory; when true, output will be saved in 
	the output file instead of displayed on the screen.

- DEV_DEBUG_CLEAN -- Minimal information of the parsing process in clear 
	messages. 

- DEV_DEBUG_CRUDE -- More parsing info in a not so readable format.

- DEV_DEBUG_STACK -- Detailed info on stack operations

- DEV_DEBUG_EXCEPTION -- Print out errors and exceptions, mostly memory 	
	handling. 	It is recommended to not compile with any DEV_DEBUG except for 	
	DEV_DEBUG_EXCEPTION if the input is more than 8 characters. The reason is 	
	that the print buffer is located on the device which might reduce 	
	performance. Unfortunately, the limitations of CUDA do not allow other 	
	alternatives for printing debugging information from the device. - 

N_STACK_SEGMENTS -- Read the Troubleshooting section before changing this.


--------------------------------------------------------------------------------

Execution:

The input string should be stored in a text file called "in" without extension.
If the input contains any characters other than a, c, g, or u, all characters
after the first hostile character will be ignored without directly
informing the user about this issue.

Run the executable file. By default, the name will be a.out
Example: ./a.out

The output will be stored in a text file named "out" by default.
--------------------------------------------------------------------------------

Output:

The output lists a summary of each iteration followed, optionally, by 
details of the actions of active cudaBlocks during that iteration.
The Iteration Summary includes:
- Threads: The number of initially active cudaBlocks, some of which were neither 
		 successful nor accepted and thus died, got killed, or committed 
		 Harakiri thereby honoring the whole run.
- Successful: Number of cudaBlocks that have taken a Shift or Reduce actions.
- Accepted:   Number of cudaBlocks that have taken an Accept action.
- Exec Time:  of the Parse() and Preprocess() functions that run on the device.
- Total Device Exec Time: is a cummulative device execution time. 

Following is a legend of all keywords and uncommon abbreviations used 
in the output:
- [x, y] -- x is the cudaBlock number. y is the thread number. Since the program
			uses one thread per block, y is expected to always be zero.
- sps -- Parse State; it is a block of data containing:
		- current state number
		- index of the character being processed
		- Status Flag: is parse successful, accepted, or error ?
		- Stack ID: ID of the stack assigned to this particular parse.
- T[x]=y : The token being processed is y at index x 
- Stt -- Current State number
- Act -- Action taken by this cudaBlock: Sx means shift and goto state x
		 Ry means Reduce by rule number y
- Nxt -- Next state or the goto-state.
- #Pops -- The number of Pops done on the stack due to a reduce operation.
- Aft -- Number of state on top of the stack after the pops of a reduce op.
- COPY Stk#x to Stk#y -- Copying My Parent's Stack#x to My Stack#y

For the output to look as neat as possible, set PRINT_TO_FILE to 1 and
make sure tab size is set to 8 for the terminal and 4 for the text editor.

Sometimes the output prints crazy numbers like state number 1769, especially
when too many cudaBlocks have been initiated and are printing. This happens
because the print buffer on the device has probably overflown. It doesn't
necessarily mean that the values are wrong, because all invalid values are
being checked in the code and taken care of.

--------------------------------------------------------------------------------

Files:

- swallow.cu
swallow.cu is the first file with which this project started.
It contains ALL the code written for this project. Later, I divided
it into:
kernel.cu
basic.cu
cudaPrintf.cu
cuStack.cu
cuDynamicArray.cu
These files are not totally independant. Every one of them depends on 
"basic.cu" : the file with all the defintions. swallow.cu is not to 
be updated anymore. I did this to make it easier to later replace 
these simple implementations of stack, dynamic array, and 
print--that were written in a hustle--with  better more stable 
classes written for cuda. cudaPrintf has already been replaced by 
the two cuPrintf files from NVIDIA. The Dynamic array can be replaced by Thrust
or Hydrazine's vector classes. 


- kernel.cu
"kernel.cu" is the file that contains the main() function and
the CUDA device kernel function. Originally, everything was in
a single file "swallow.cu" but I thought it would be cleaner to divide

- y.output
The file "y.output" is not required. It is the file generated by bison given
the grammar. It shows all states in the state space and all shift/reduce
and reduce/reduce conflicts.

--------------------------------------------------------------------------------

Troubleshooting:

- When parsing an input longer than 25 characters, a memory hog might occur
  because of the technical limitations of CUDA.
- In the program ouput, if the number of stack segments actually used is 
  equal to the number of reserved segments, then this is an indication that
  program might have killed possible parses to free memory. A constant
  defined early in the code "N_STACK_SEGMENTS" should be increased to
  cover the need.
- For stack invalid operations, the program should handle them automatically
  and kill error-causing threads to continue execution.
- Generally, Errors and Exceptions will be printed out to the user with
  instructions on how to avoid.
  
--------------------------------------------------------------------------------

Contact:
For any issues or suggestions, please contact: 3omarz+swallow@gmail.com
3 Feb 2012

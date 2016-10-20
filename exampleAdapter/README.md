# C++/Python Wrapping Example with SWIG
	
	This code is an example of how some stateful objects in C++ can be wrapped

## Dependancies

 1. You will need to have swig installed. I recomend installing cygwin and adding the swig library
 1. premake4 is given in this code
 1. Visual Studio 2013
 1. Keras + Theano (for the cool example)
 
 
## Building

 1. run gen_swig_example.sh
  1. This generates the SWIG wrapper code that should not be commited to any versioning system.
 1. run premake4.exe vs2012
  1. This generates a visual studio project for the code.
  1. You might need to update the path to your installation of Python.
 1. Open the example.sln file
 1. Retarget the solution to your version of visual studio if need be.
 1. In the example project settings remove Import library (Configuration Properties-> Linker -> All Options -> Import Library). Delete the contents of this field.
 1. Compile a Release64 version of the code.
  1. More steps might be nessicary for a debug version of the code.


## Gotcha's 

 1. Python Will garbage collect your objects! Be sure to keep refernces to objects you don't want garabage collected
 2. There can be some confusion between Python types nympy types and C++ types. For example, numpy.float32 != (float() == C++ double).
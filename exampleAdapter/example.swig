

 /* example.i */
 /*
 Read http://www.swig.org/papers/PyTutorial98/PyTutorial98.pdf for more info
 */
 /* THe name of the module is important and must match the library name */
 %module example
 %{
 /* Put header files here or function declarations like below */
 #define SWIG_FILE_WITH_INIT
 #include <vector>
 #include "include/example.h"
 #include "include/Word.h"
 #include "include/Function.h"
 %}
 
 /* These are the file to export in the library */
 %include "std_string.i"
 %include "std_vector.i"
 %include "std_map.i"
 %include "std_pair.i"
 // %include "std_set.i"
 
 // Instantiate templates used by example
namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   // If you want to be able to use nested vectors....
   %template(DoubleVectorVector) vector<vector<double>>;
   %template(StringVector) vector<string>;
   %template(Vector_SizeT) vector<size_t>;
   %template(FloatVector) vector<float>;
   // %template(SizeTSet) set<size_t>;
}

 %include "include/Word.h"
 %include "include/Function.h"
 %include "include/example.h"
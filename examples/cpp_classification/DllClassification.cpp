  #include "DllClassification.h"
  #include "Classification.h"
  DllClassification *CaffeClassification(const std::string &model_file,const std::string &train_file){	
	  return (DllClassification *)new Classification(model_file, train_file);
  }

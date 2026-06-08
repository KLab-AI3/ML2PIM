add_mlir_conversion_library(MLIRLLVMPIMToLLVM
  LLVMPIMToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LLVMPIMToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRLLVMPIMDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRTransforms
  )

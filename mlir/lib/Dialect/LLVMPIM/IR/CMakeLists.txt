add_mlir_dialect_library(MLIRLLVMPIMDialect
  LLVMPIMDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMPIM

  DEPENDS
  MLIRLLVMPIMOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  )

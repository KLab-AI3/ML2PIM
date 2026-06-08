#ifndef MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVMPIMPASS_H_
#define MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVMPIMPASS_H_

#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
    // class DialectRegistry;
    class LLVMTypeConverter;
    // class RewritePatternSet;
    // class Pass;
    
    std::unique_ptr<Pass> createConvertVectorToLLVMPIMPass();
} // namespace mlir

namespace mlir {
class Pass;

#define GEN_PASS_DECL_CONVERTVECTORTOLLVMPIMPASS  //GEN_PASS_DECL , GEN_PASS_DEF
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir



#endif // MLIR_CONVERSION_VECTORTOLLVM_VECTORTOLLVMPIMPASS_H
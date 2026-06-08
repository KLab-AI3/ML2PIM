//===- LLVMPIMToLLVM.h - Convert LLVMPIM to LLVM dialect -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LLVMPIMTOLLVM_LLVMPIMTOLLVM_H
#define MLIR_CONVERSION_LLVMPIMTOLLVM_LLVMPIMTOLLVM_H

#include "mlir/IR/PatternMatch.h"
#include <memory>

namespace mlir {

class LLVMTypeConverter;
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTLLVMPIMTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"

void populateLLVMPIMToLLVMConversionPatterns(const LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_LLVMPIMTOLLVM_LLVMPIMTOLLVM_H

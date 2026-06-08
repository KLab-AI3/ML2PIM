//===- LLVMPIMDialect.cpp - MLIR LLVMPIM dialect -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMPIM/IR/LLVMPIMOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::llvmpim;

#include "mlir/Dialect/LLVMPIM/IR/LLVMPIMOpsDialect.cpp.inc"

void LLVMPIMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMPIM/IR/LLVMPIMOps.cpp.inc"
      >();
}

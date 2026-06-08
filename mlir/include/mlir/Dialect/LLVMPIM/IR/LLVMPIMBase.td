//===- LLVMPIMBase.td - LLVMPIM dialect base definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVMPIM_BASE
#define LLVMPIM_BASE

include "mlir/IR/OpBase.td"

def LLVMPIM_Dialect : Dialect {
  let name = "llvmpim";
  let cppNamespace = "::mlir::llvmpim";
  let dependentDialects = [
    "::mlir::memref::MemRefDialect"
  ];
  let description = [{
    The LLVMPIM dialect models row-level processing-in-memory operations before
    they are lowered to LLVM dialect calls into a PIM runtime.
  }];
}

#endif // LLVMPIM_BASE

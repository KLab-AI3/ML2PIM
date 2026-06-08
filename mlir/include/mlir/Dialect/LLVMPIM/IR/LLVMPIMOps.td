//===- LLVMPIMOps.td - LLVMPIM operation definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVMPIM_OPS
#define LLVMPIM_OPS

include "mlir/Dialect/LLVMPIM/IR/LLVMPIMBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class LLVMPIM_Op<string mnemonic, list<Trait> traits = []> :
    Op<LLVMPIM_Dialect, mnemonic, traits>;

class LLVMPIM_RowBinaryOp<string mnemonic, string summaryStr> :
    LLVMPIM_Op<mnemonic, [
      AllTypesMatch<["lhs", "rhs", "out"]>,
      MemoryEffects<[MemReadAt<0, FullEffect>, MemReadAt<1, FullEffect>,
                     MemWriteAt<2, FullEffect>]>
    ]> {
  let summary = summaryStr;
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$out);
  let assemblyFormat =
      "$lhs `,` $rhs `,` $out attr-dict `:` type($lhs)";
}

class LLVMPIM_RowUnaryOp<string mnemonic, string summaryStr> :
    LLVMPIM_Op<mnemonic, [
      AllTypesMatch<["input", "out"]>,
      MemoryEffects<[MemReadAt<0, FullEffect>, MemWriteAt<1, FullEffect>]>
    ]> {
  let summary = summaryStr;
  let arguments = (ins AnyMemRef:$input, AnyMemRef:$out);
  let assemblyFormat =
      "$input `,` $out attr-dict `:` type($input)";
}

def RowOpAdd4 : LLVMPIM_RowBinaryOp<"rowop_add4", "PIM row add operation">;
def RowOpAdd8 : LLVMPIM_RowBinaryOp<"rowop_add8", "PIM row add operation">;
def RowOpAdd16 : LLVMPIM_RowBinaryOp<"rowop_add16", "PIM row add operation">;
def RowOpAdd32 : LLVMPIM_RowBinaryOp<"rowop_add32", "PIM row add operation">;
def RowOpAddf16
    : LLVMPIM_RowBinaryOp<"rowop_addf16", "PIM row add operation">;

def RowOpMult4
    : LLVMPIM_RowBinaryOp<"rowop_mult4", "PIM row multiply operation">;
def RowOpMult8
    : LLVMPIM_RowBinaryOp<"rowop_mult8", "PIM row multiply operation">;
def RowOpMult16
    : LLVMPIM_RowBinaryOp<"rowop_mult16", "PIM row multiply operation">;
def RowOpMult32
    : LLVMPIM_RowBinaryOp<"rowop_mult32", "PIM row multiply operation">;
def RowOpMultf16
    : LLVMPIM_RowBinaryOp<"rowop_multf16", "PIM row multiply operation">;

def RowOpSub4 : LLVMPIM_RowBinaryOp<"rowop_sub4", "PIM row subtract operation">;
def RowOpSub8 : LLVMPIM_RowBinaryOp<"rowop_sub8", "PIM row subtract operation">;
def RowOpSub16
    : LLVMPIM_RowBinaryOp<"rowop_sub16", "PIM row subtract operation">;
def RowOpSub32
    : LLVMPIM_RowBinaryOp<"rowop_sub32", "PIM row subtract operation">;
def RowOpSubf16
    : LLVMPIM_RowBinaryOp<"rowop_subf16", "PIM row subtract operation">;

def RowOpDiv4 : LLVMPIM_RowBinaryOp<"rowop_div4", "PIM row divide operation">;
def RowOpDiv8 : LLVMPIM_RowBinaryOp<"rowop_div8", "PIM row divide operation">;
def RowOpDiv16
    : LLVMPIM_RowBinaryOp<"rowop_div16", "PIM row divide operation">;
def RowOpDiv32
    : LLVMPIM_RowBinaryOp<"rowop_div32", "PIM row divide operation">;
def RowOpDivf16
    : LLVMPIM_RowBinaryOp<"rowop_divf16", "PIM row divide operation">;

def RowOpGe4
    : LLVMPIM_RowBinaryOp<"rowop_ge4",
                          "PIM row greater-or-equal comparison operation">;
def RowOpGe8
    : LLVMPIM_RowBinaryOp<"rowop_ge8",
                          "PIM row greater-or-equal comparison operation">;
def RowOpGe16
    : LLVMPIM_RowBinaryOp<"rowop_ge16",
                          "PIM row greater-or-equal comparison operation">;
def RowOpGe32
    : LLVMPIM_RowBinaryOp<"rowop_ge32",
                          "PIM row greater-or-equal comparison operation">;
def RowOpGef16
    : LLVMPIM_RowBinaryOp<"rowop_gef16",
                          "PIM row greater-or-equal comparison operation">;

def RowOpLe4
    : LLVMPIM_RowBinaryOp<"rowop_le4",
                          "PIM row less-or-equal comparison operation">;
def RowOpLe8
    : LLVMPIM_RowBinaryOp<"rowop_le8",
                          "PIM row less-or-equal comparison operation">;
def RowOpLe16
    : LLVMPIM_RowBinaryOp<"rowop_le16",
                          "PIM row less-or-equal comparison operation">;
def RowOpLe32
    : LLVMPIM_RowBinaryOp<"rowop_le32",
                          "PIM row less-or-equal comparison operation">;
def RowOpLef16
    : LLVMPIM_RowBinaryOp<"rowop_lef16",
                          "PIM row less-or-equal comparison operation">;

def RowOpRelu4 : LLVMPIM_RowUnaryOp<"rowop_relu4", "PIM row ReLU operation">;
def RowOpRelu8 : LLVMPIM_RowUnaryOp<"rowop_relu8", "PIM row ReLU operation">;
def RowOpRelu16
    : LLVMPIM_RowUnaryOp<"rowop_relu16", "PIM row ReLU operation">;
def RowOpRelu32
    : LLVMPIM_RowUnaryOp<"rowop_relu32", "PIM row ReLU operation">;
def RowOpReluf16
    : LLVMPIM_RowUnaryOp<"rowop_reluf16", "PIM row ReLU operation">;

def RowOpSoftmax4
    : LLVMPIM_RowUnaryOp<"rowop_softmax4", "PIM row softmax operation">;
def RowOpSoftmax8
    : LLVMPIM_RowUnaryOp<"rowop_softmax8", "PIM row softmax operation">;
def RowOpSoftmax16
    : LLVMPIM_RowUnaryOp<"rowop_softmax16", "PIM row softmax operation">;
def RowOpSoftmax32
    : LLVMPIM_RowUnaryOp<"rowop_softmax32", "PIM row softmax operation">;
def RowOpSoftmaxf16
    : LLVMPIM_RowUnaryOp<"rowop_softmaxf16", "PIM row softmax operation">;

def RowOpSin4 : LLVMPIM_RowUnaryOp<"rowop_sin4", "PIM row sine operation">;
def RowOpSin8 : LLVMPIM_RowUnaryOp<"rowop_sin8", "PIM row sine operation">;
def RowOpSin16 : LLVMPIM_RowUnaryOp<"rowop_sin16", "PIM row sine operation">;
def RowOpSin32 : LLVMPIM_RowUnaryOp<"rowop_sin32", "PIM row sine operation">;
def RowOpSinf16
    : LLVMPIM_RowUnaryOp<"rowop_sinf16", "PIM row sine operation">;

def RowOpCos4 : LLVMPIM_RowUnaryOp<"rowop_cos4", "PIM row cosine operation">;
def RowOpCos8 : LLVMPIM_RowUnaryOp<"rowop_cos8", "PIM row cosine operation">;
def RowOpCos16
    : LLVMPIM_RowUnaryOp<"rowop_cos16", "PIM row cosine operation">;
def RowOpCos32
    : LLVMPIM_RowUnaryOp<"rowop_cos32", "PIM row cosine operation">;
def RowOpCosf16
    : LLVMPIM_RowUnaryOp<"rowop_cosf16", "PIM row cosine operation">;

def RowOpAtan4 : LLVMPIM_RowUnaryOp<"rowop_atan4", "PIM row atan operation">;
def RowOpAtan8 : LLVMPIM_RowUnaryOp<"rowop_atan8", "PIM row atan operation">;
def RowOpAtan16
    : LLVMPIM_RowUnaryOp<"rowop_atan16", "PIM row atan operation">;
def RowOpAtan32
    : LLVMPIM_RowUnaryOp<"rowop_atan32", "PIM row atan operation">;
def RowOpAtanf16
    : LLVMPIM_RowUnaryOp<"rowop_atanf16", "PIM row atan operation">;

def RowOpSinh4
    : LLVMPIM_RowUnaryOp<"rowop_sinh4", "PIM row hyperbolic sine operation">;
def RowOpSinh8
    : LLVMPIM_RowUnaryOp<"rowop_sinh8", "PIM row hyperbolic sine operation">;
def RowOpSinh16
    : LLVMPIM_RowUnaryOp<"rowop_sinh16", "PIM row hyperbolic sine operation">;
def RowOpSinh32
    : LLVMPIM_RowUnaryOp<"rowop_sinh32", "PIM row hyperbolic sine operation">;
def RowOpSinhf16
    : LLVMPIM_RowUnaryOp<"rowop_sinhf16", "PIM row hyperbolic sine operation">;

def RowOpCosh4
    : LLVMPIM_RowUnaryOp<"rowop_cosh4", "PIM row hyperbolic cosine operation">;
def RowOpCosh8
    : LLVMPIM_RowUnaryOp<"rowop_cosh8", "PIM row hyperbolic cosine operation">;
def RowOpCosh16
    : LLVMPIM_RowUnaryOp<"rowop_cosh16", "PIM row hyperbolic cosine operation">;
def RowOpCosh32
    : LLVMPIM_RowUnaryOp<"rowop_cosh32", "PIM row hyperbolic cosine operation">;
def RowOpCoshf16
    : LLVMPIM_RowUnaryOp<"rowop_coshf16", "PIM row hyperbolic cosine operation">;

def RowOpAtanh4
    : LLVMPIM_RowUnaryOp<"rowop_atanh4", "PIM row hyperbolic atan operation">;
def RowOpAtanh8
    : LLVMPIM_RowUnaryOp<"rowop_atanh8", "PIM row hyperbolic atan operation">;
def RowOpAtanh16
    : LLVMPIM_RowUnaryOp<"rowop_atanh16", "PIM row hyperbolic atan operation">;
def RowOpAtanh32
    : LLVMPIM_RowUnaryOp<"rowop_atanh32", "PIM row hyperbolic atan operation">;
def RowOpAtanhf16
    : LLVMPIM_RowUnaryOp<"rowop_atanhf16", "PIM row hyperbolic atan operation">;

def RowOpSqrt4
    : LLVMPIM_RowUnaryOp<"rowop_sqrt4", "PIM row square root operation">;
def RowOpSqrt8
    : LLVMPIM_RowUnaryOp<"rowop_sqrt8", "PIM row square root operation">;
def RowOpSqrt16
    : LLVMPIM_RowUnaryOp<"rowop_sqrt16", "PIM row square root operation">;
def RowOpSqrt32
    : LLVMPIM_RowUnaryOp<"rowop_sqrt32", "PIM row square root operation">;
def RowOpSqrtf16
    : LLVMPIM_RowUnaryOp<"rowop_sqrtf16", "PIM row square root operation">;

def RowOpMaxPool8 : LLVMPIM_RowBinaryOp<"rowop_maxpool8",
                                          "PIM row maxpool operation">;

#endif // LLVMPIM_OPS

//===- LLVMPIMToLLVM.cpp - Convert LLVMPIM to LLVM dialect ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMPIMToLLVM/LLVMPIMToLLVM.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMPIM/IR/LLVMPIMOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLLVMPIMTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

static Value getAlignedMemRefPtr(Location loc, Value memrefDescriptor,
                                 ConversionPatternRewriter &rewriter) {
  return MemRefDescriptor(memrefDescriptor).alignedPtr(rewriter, loc);
}

static FailureOr<LLVM::LLVMFuncOp>
lookupOrCreateRowOpFn(ConversionPatternRewriter &rewriter, Operation *op,
                      StringRef functionName, unsigned numOperands) {
  auto moduleOp = op->getParentWithTrait<OpTrait::SymbolTable>();
  if (!moduleOp) {
    op->emitError("expected parent operation with symbol table");
    return failure();
  }

  Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  SmallVector<Type> argTypes(numOperands, ptrTy);
  Type voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
  return LLVM::lookupOrCreateFn(rewriter, moduleOp, functionName, argTypes,
                                voidTy);
}

namespace {
template <typename SourceOp>
struct RowOpCallLowering : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  RowOpCallLowering(const LLVMTypeConverter &converter, StringRef functionName)
      : OpConversionPattern<SourceOp>(converter, &converter.getContext()),
        functionName(functionName.str()) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> callOperands;
    callOperands.reserve(adaptor.getOperands().size());
    for (Value operand : adaptor.getOperands())
      callOperands.push_back(getAlignedMemRefPtr(op.getLoc(), operand,
                                                 rewriter));

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        lookupOrCreateRowOpFn(rewriter, op.getOperation(), functionName,
                              callOperands.size());
    if (failed(funcOp))
      return failure();

    LLVM::CallOp::create(rewriter, op.getLoc(), *funcOp, callOperands);
    rewriter.eraseOp(op);
    return success();
  }

private:
  std::string functionName;
};

struct ConvertLLVMPIMToLLVMPass
    : public impl::ConvertLLVMPIMToLLVMPassBase<
          ConvertLLVMPIMToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    MLIRContext &context = getContext();
    LLVMTypeConverter converter(&context);
    RewritePatternSet patterns(&context);
    populateLLVMPIMToLLVMConversionPatterns(converter, patterns);

    ConversionTarget target(context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<llvmpim::LLVMPIMDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::populateLLVMPIMToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<RowOpCallLowering<llvmpim::RowOpAdd4>>(converter,
                                                        "rowop_add4");
  patterns.add<RowOpCallLowering<llvmpim::RowOpAdd8>>(converter,
                                                        "rowop_add8");
  patterns.add<RowOpCallLowering<llvmpim::RowOpAdd16>>(converter,
                                                         "rowop_add16");
  patterns.add<RowOpCallLowering<llvmpim::RowOpAdd32>>(converter,
                                                         "rowop_add32");
  patterns.add<RowOpCallLowering<llvmpim::RowOpAddf16>>(converter,
                                                          "rowop_addf16");

  patterns.add<RowOpCallLowering<llvmpim::RowOpMult4>>(converter,
                                                         "rowop_mult4");
  patterns.add<RowOpCallLowering<llvmpim::RowOpMult8>>(converter,
                                                         "rowop_mult8");
  patterns.add<RowOpCallLowering<llvmpim::RowOpMult16>>(converter,
                                                          "rowop_mult16");
  patterns.add<RowOpCallLowering<llvmpim::RowOpMult32>>(converter,
                                                          "rowop_mult32");
  patterns.add<RowOpCallLowering<llvmpim::RowOpMultf16>>(converter,
                                                           "rowop_multf16");

  patterns.add<RowOpCallLowering<llvmpim::RowOpSub4>>(converter,
                                                        "rowop_sub4");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSub8>>(converter,
                                                        "rowop_sub8");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSub16>>(converter,
                                                         "rowop_sub16");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSub32>>(converter,
                                                         "rowop_sub32");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSubf16>>(converter,
                                                          "rowop_subf16");

  patterns.add<RowOpCallLowering<llvmpim::RowOpDiv4>>(converter,
                                                        "rowop_div4");
  patterns.add<RowOpCallLowering<llvmpim::RowOpDiv8>>(converter,
                                                        "rowop_div8");
  patterns.add<RowOpCallLowering<llvmpim::RowOpDiv16>>(converter,
                                                         "rowop_div16");
  patterns.add<RowOpCallLowering<llvmpim::RowOpDiv32>>(converter,
                                                         "rowop_div32");
  patterns.add<RowOpCallLowering<llvmpim::RowOpDivf16>>(converter,
                                                          "rowop_divf16");

  patterns.add<RowOpCallLowering<llvmpim::RowOpRelu4>>(converter,
                                                         "rowop_relu4");
  patterns.add<RowOpCallLowering<llvmpim::RowOpRelu8>>(converter,
                                                         "rowop_relu8");
  patterns.add<RowOpCallLowering<llvmpim::RowOpRelu16>>(converter,
                                                          "rowop_relu16");
  patterns.add<RowOpCallLowering<llvmpim::RowOpRelu32>>(converter,
                                                          "rowop_relu32");
  patterns.add<RowOpCallLowering<llvmpim::RowOpReluf16>>(converter,
                                                           "rowop_reluf16");

  patterns.add<RowOpCallLowering<llvmpim::RowOpGe4>>(converter,
                                                       "rowop_ge4");
  patterns.add<RowOpCallLowering<llvmpim::RowOpGe8>>(converter,
                                                       "rowop_ge8");
  patterns.add<RowOpCallLowering<llvmpim::RowOpGe16>>(converter,
                                                        "rowop_ge16");
  patterns.add<RowOpCallLowering<llvmpim::RowOpGe32>>(converter,
                                                        "rowop_ge32");
  patterns.add<RowOpCallLowering<llvmpim::RowOpGef16>>(converter,
                                                         "rowop_gef16");

  patterns.add<RowOpCallLowering<llvmpim::RowOpLe4>>(converter,
                                                       "rowop_le4");
  patterns.add<RowOpCallLowering<llvmpim::RowOpLe8>>(converter,
                                                       "rowop_le8");
  patterns.add<RowOpCallLowering<llvmpim::RowOpLe16>>(converter,
                                                        "rowop_le16");
  patterns.add<RowOpCallLowering<llvmpim::RowOpLe32>>(converter,
                                                        "rowop_le32");
  patterns.add<RowOpCallLowering<llvmpim::RowOpLef16>>(converter,
                                                         "rowop_lef16");

  patterns.add<RowOpCallLowering<llvmpim::RowOpSoftmax4>>(
      converter, "rowop_softmax4");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSoftmax8>>(
      converter, "rowop_softmax8");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSoftmax16>>(
      converter, "rowop_softmax16");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSoftmax32>>(
      converter, "rowop_softmax32");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSoftmaxf16>>(
      converter, "rowop_softmaxf16");

  patterns.add<RowOpCallLowering<llvmpim::RowOpSin4>>(converter,
                                                        "rowop_sin4");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSin8>>(converter,
                                                        "rowop_sin8");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSin16>>(converter,
                                                         "rowop_sin16");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSin32>>(converter,
                                                         "rowop_sin32");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSinf16>>(converter,
                                                          "rowop_sinf16");

  patterns.add<RowOpCallLowering<llvmpim::RowOpCos4>>(converter,
                                                        "rowop_cos4");
  patterns.add<RowOpCallLowering<llvmpim::RowOpCos8>>(converter,
                                                        "rowop_cos8");
  patterns.add<RowOpCallLowering<llvmpim::RowOpCos16>>(converter,
                                                         "rowop_cos16");
  patterns.add<RowOpCallLowering<llvmpim::RowOpCos32>>(converter,
                                                         "rowop_cos32");
  patterns.add<RowOpCallLowering<llvmpim::RowOpCosf16>>(converter,
                                                          "rowop_cosf16");

  patterns.add<RowOpCallLowering<llvmpim::RowOpAtan4>>(converter,
                                                         "rowop_atan4");
  patterns.add<RowOpCallLowering<llvmpim::RowOpAtan8>>(converter,
                                                         "rowop_atan8");
  patterns.add<RowOpCallLowering<llvmpim::RowOpAtan16>>(converter,
                                                          "rowop_atan16");
  patterns.add<RowOpCallLowering<llvmpim::RowOpAtan32>>(converter,
                                                          "rowop_atan32");
  patterns.add<RowOpCallLowering<llvmpim::RowOpAtanf16>>(converter,
                                                           "rowop_atanf16");

  patterns.add<RowOpCallLowering<llvmpim::RowOpSinh4>>(converter,
                                                         "rowop_sinh4");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSinh8>>(converter,
                                                         "rowop_sinh8");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSinh16>>(converter,
                                                          "rowop_sinh16");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSinh32>>(converter,
                                                          "rowop_sinh32");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSinhf16>>(converter,
                                                           "rowop_sinhf16");

  patterns.add<RowOpCallLowering<llvmpim::RowOpCosh4>>(converter,
                                                         "rowop_cosh4");
  patterns.add<RowOpCallLowering<llvmpim::RowOpCosh8>>(converter,
                                                         "rowop_cosh8");
  patterns.add<RowOpCallLowering<llvmpim::RowOpCosh16>>(converter,
                                                          "rowop_cosh16");
  patterns.add<RowOpCallLowering<llvmpim::RowOpCosh32>>(converter,
                                                          "rowop_cosh32");
  patterns.add<RowOpCallLowering<llvmpim::RowOpCoshf16>>(converter,
                                                           "rowop_coshf16");

  patterns.add<RowOpCallLowering<llvmpim::RowOpAtanh4>>(converter,
                                                          "rowop_atanh4");
  patterns.add<RowOpCallLowering<llvmpim::RowOpAtanh8>>(converter,
                                                          "rowop_atanh8");
  patterns.add<RowOpCallLowering<llvmpim::RowOpAtanh16>>(converter,
                                                           "rowop_atanh16");
  patterns.add<RowOpCallLowering<llvmpim::RowOpAtanh32>>(converter,
                                                           "rowop_atanh32");
  patterns.add<RowOpCallLowering<llvmpim::RowOpAtanhf16>>(
      converter, "rowop_atanhf16");

  patterns.add<RowOpCallLowering<llvmpim::RowOpSqrt4>>(converter,
                                                         "rowop_sqrt4");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSqrt8>>(converter,
                                                         "rowop_sqrt8");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSqrt16>>(converter,
                                                          "rowop_sqrt16");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSqrt32>>(converter,
                                                          "rowop_sqrt32");
  patterns.add<RowOpCallLowering<llvmpim::RowOpSqrtf16>>(converter,
                                                           "rowop_sqrtf16");

  patterns.add<RowOpCallLowering<llvmpim::RowOpMaxPool8>>(converter,
                                                            "rowop_maxpool8");
}

//===- ConvertVectorToLLVMPIMPass.cpp -------------------------------------===//
//
// Custom pass: Vector + Arith patterns -> LLVMPIM row operations,
// then lower remaining Vector ops to LLVM without leaving stray load/store.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPIMPass.h"

#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/PrintCallHelper.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMPIM/IR/LLVMPIMOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Casting.h"

#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTVECTORTOLLVMPIMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::vector;

// ============================================================================
// PIM pattern helpers
// ============================================================================

enum class RowDataType { I4, I8, I16, I32, F16 };
enum class RowBinaryKind { Add, Mult, Sub, Div, Ge, Le };
enum class RowUnaryKind {
  Relu,
  Softmax,
  Sin,
  Cos,
  Atan,
  Sinh,
  Cosh,
  Atanh,
  Sqrt
};

// Check that the vector length is supported by the row hardware path.
static bool isSupportedVector(Value v) {
  auto vt = dyn_cast<VectorType>(v.getType());
  if (!vt)
    return false;
  return vt.getNumElements() <= 8192;
}

static std::optional<RowDataType> getRowDataType(Type type) {
  Type elementType = getElementTypeOrSelf(type);
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case 4:
      return RowDataType::I4;
    case 8:
      return RowDataType::I8;
    case 16:
      return RowDataType::I16;
    case 32:
      return RowDataType::I32;
    default:
      return std::nullopt;
    }
  }
  if (elementType.isF16())
    return RowDataType::F16;
  return std::nullopt;
}

static std::optional<RowDataType> getRowDataTypeFromMemRef(Value value) {
  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType)
    return std::nullopt;
  return getRowDataType(memrefType.getElementType());
}

static bool haveSameMemRefType(Value lhs, Value rhs, Value out) {
  auto lhsType = dyn_cast<MemRefType>(lhs.getType());
  auto rhsType = dyn_cast<MemRefType>(rhs.getType());
  auto outType = dyn_cast<MemRefType>(out.getType());
  return lhsType && rhsType && outType && lhsType == rhsType &&
         lhsType == outType;
}

static bool haveSameMemRefType(Value input, Value out) {
  auto inputType = dyn_cast<MemRefType>(input.getType());
  auto outType = dyn_cast<MemRefType>(out.getType());
  return inputType && outType && inputType == outType;
}

template <typename Op4, typename Op8, typename Op16, typename Op32,
          typename OpF16>
static LogicalResult createTypedRowBinaryOp(PatternRewriter &rewriter,
                                            Location loc,
                                            RowDataType dataType, Value lhs,
                                            Value rhs, Value out) {
  switch (dataType) {
  case RowDataType::I4:
    rewriter.create<Op4>(loc, lhs, rhs, out);
    return success();
  case RowDataType::I8:
    rewriter.create<Op8>(loc, lhs, rhs, out);
    return success();
  case RowDataType::I16:
    rewriter.create<Op16>(loc, lhs, rhs, out);
    return success();
  case RowDataType::I32:
    rewriter.create<Op32>(loc, lhs, rhs, out);
    return success();
  case RowDataType::F16:
    rewriter.create<OpF16>(loc, lhs, rhs, out);
    return success();
  }
  return failure();
}

template <typename Op4, typename Op8, typename Op16, typename Op32,
          typename OpF16>
static LogicalResult createTypedRowUnaryOp(PatternRewriter &rewriter,
                                           Location loc,
                                           RowDataType dataType, Value input,
                                           Value out) {
  switch (dataType) {
  case RowDataType::I4:
    rewriter.create<Op4>(loc, input, out);
    return success();
  case RowDataType::I8:
    rewriter.create<Op8>(loc, input, out);
    return success();
  case RowDataType::I16:
    rewriter.create<Op16>(loc, input, out);
    return success();
  case RowDataType::I32:
    rewriter.create<Op32>(loc, input, out);
    return success();
  case RowDataType::F16:
    rewriter.create<OpF16>(loc, input, out);
    return success();
  }
  return failure();
}

static LogicalResult createRowBinaryOp(PatternRewriter &rewriter, Location loc,
                                       RowBinaryKind kind,
                                       RowDataType dataType, Value lhs,
                                       Value rhs, Value out) {
  switch (kind) {
  case RowBinaryKind::Add:
    return createTypedRowBinaryOp<llvmpim::RowOpAdd4, llvmpim::RowOpAdd8,
                                  llvmpim::RowOpAdd16, llvmpim::RowOpAdd32,
                                  llvmpim::RowOpAddf16>(
        rewriter, loc, dataType, lhs, rhs, out);
  case RowBinaryKind::Mult:
    return createTypedRowBinaryOp<llvmpim::RowOpMult4, llvmpim::RowOpMult8,
                                  llvmpim::RowOpMult16, llvmpim::RowOpMult32,
                                  llvmpim::RowOpMultf16>(
        rewriter, loc, dataType, lhs, rhs, out);
  case RowBinaryKind::Sub:
    return createTypedRowBinaryOp<llvmpim::RowOpSub4, llvmpim::RowOpSub8,
                                  llvmpim::RowOpSub16, llvmpim::RowOpSub32,
                                  llvmpim::RowOpSubf16>(
        rewriter, loc, dataType, lhs, rhs, out);
  case RowBinaryKind::Div:
    return createTypedRowBinaryOp<llvmpim::RowOpDiv4, llvmpim::RowOpDiv8,
                                  llvmpim::RowOpDiv16, llvmpim::RowOpDiv32,
                                  llvmpim::RowOpDivf16>(
        rewriter, loc, dataType, lhs, rhs, out);
  case RowBinaryKind::Ge:
    return createTypedRowBinaryOp<llvmpim::RowOpGe4, llvmpim::RowOpGe8,
                                  llvmpim::RowOpGe16, llvmpim::RowOpGe32,
                                  llvmpim::RowOpGef16>(
        rewriter, loc, dataType, lhs, rhs, out);
  case RowBinaryKind::Le:
    return createTypedRowBinaryOp<llvmpim::RowOpLe4, llvmpim::RowOpLe8,
                                  llvmpim::RowOpLe16, llvmpim::RowOpLe32,
                                  llvmpim::RowOpLef16>(
        rewriter, loc, dataType, lhs, rhs, out);
  }
  return failure();
}

static LogicalResult createRowUnaryOp(PatternRewriter &rewriter, Location loc,
                                      RowUnaryKind kind, RowDataType dataType,
                                      Value input, Value out) {
  switch (kind) {
  case RowUnaryKind::Relu:
    return createTypedRowUnaryOp<llvmpim::RowOpRelu4, llvmpim::RowOpRelu8,
                                 llvmpim::RowOpRelu16, llvmpim::RowOpRelu32,
                                 llvmpim::RowOpReluf16>(
        rewriter, loc, dataType, input, out);
  case RowUnaryKind::Softmax:
    return createTypedRowUnaryOp<llvmpim::RowOpSoftmax4,
                                 llvmpim::RowOpSoftmax8,
                                 llvmpim::RowOpSoftmax16,
                                 llvmpim::RowOpSoftmax32,
                                 llvmpim::RowOpSoftmaxf16>(
        rewriter, loc, dataType, input, out);
  case RowUnaryKind::Sin:
    return createTypedRowUnaryOp<llvmpim::RowOpSin4, llvmpim::RowOpSin8,
                                 llvmpim::RowOpSin16, llvmpim::RowOpSin32,
                                 llvmpim::RowOpSinf16>(
        rewriter, loc, dataType, input, out);
  case RowUnaryKind::Cos:
    return createTypedRowUnaryOp<llvmpim::RowOpCos4, llvmpim::RowOpCos8,
                                 llvmpim::RowOpCos16, llvmpim::RowOpCos32,
                                 llvmpim::RowOpCosf16>(
        rewriter, loc, dataType, input, out);
  case RowUnaryKind::Atan:
    return createTypedRowUnaryOp<llvmpim::RowOpAtan4, llvmpim::RowOpAtan8,
                                 llvmpim::RowOpAtan16, llvmpim::RowOpAtan32,
                                 llvmpim::RowOpAtanf16>(
        rewriter, loc, dataType, input, out);
  case RowUnaryKind::Sinh:
    return createTypedRowUnaryOp<llvmpim::RowOpSinh4, llvmpim::RowOpSinh8,
                                 llvmpim::RowOpSinh16, llvmpim::RowOpSinh32,
                                 llvmpim::RowOpSinhf16>(
        rewriter, loc, dataType, input, out);
  case RowUnaryKind::Cosh:
    return createTypedRowUnaryOp<llvmpim::RowOpCosh4, llvmpim::RowOpCosh8,
                                 llvmpim::RowOpCosh16, llvmpim::RowOpCosh32,
                                 llvmpim::RowOpCoshf16>(
        rewriter, loc, dataType, input, out);
  case RowUnaryKind::Atanh:
    return createTypedRowUnaryOp<llvmpim::RowOpAtanh4, llvmpim::RowOpAtanh8,
                                 llvmpim::RowOpAtanh16, llvmpim::RowOpAtanh32,
                                 llvmpim::RowOpAtanhf16>(
        rewriter, loc, dataType, input, out);
  case RowUnaryKind::Sqrt:
    return createTypedRowUnaryOp<llvmpim::RowOpSqrt4, llvmpim::RowOpSqrt8,
                                 llvmpim::RowOpSqrt16, llvmpim::RowOpSqrt32,
                                 llvmpim::RowOpSqrtf16>(
        rewriter, loc, dataType, input, out);
  }
  return failure();
}

static StringRef getRowBinaryName(RowBinaryKind kind) {
  switch (kind) {
  case RowBinaryKind::Add:
    return "add";
  case RowBinaryKind::Mult:
    return "mult";
  case RowBinaryKind::Sub:
    return "sub";
  case RowBinaryKind::Div:
    return "div";
  case RowBinaryKind::Ge:
    return "ge";
  case RowBinaryKind::Le:
    return "le";
  }
  return "unknown";
}

static StringRef getRowUnaryName(RowUnaryKind kind) {
  switch (kind) {
  case RowUnaryKind::Relu:
    return "relu";
  case RowUnaryKind::Softmax:
    return "softmax";
  case RowUnaryKind::Sin:
    return "sin";
  case RowUnaryKind::Cos:
    return "cos";
  case RowUnaryKind::Atan:
    return "atan";
  case RowUnaryKind::Sinh:
    return "sinh";
  case RowUnaryKind::Cosh:
    return "cosh";
  case RowUnaryKind::Atanh:
    return "atanh";
  case RowUnaryKind::Sqrt:
    return "sqrt";
  }
  return "unknown";
}

static bool isIntegerGePredicate(arith::CmpIPredicate predicate) {
  return predicate == arith::CmpIPredicate::sge ||
         predicate == arith::CmpIPredicate::uge;
}

static bool isIntegerLePredicate(arith::CmpIPredicate predicate) {
  return predicate == arith::CmpIPredicate::sle ||
         predicate == arith::CmpIPredicate::ule;
}

static bool isFloatGePredicate(arith::CmpFPredicate predicate) {
  return predicate == arith::CmpFPredicate::OGE ||
         predicate == arith::CmpFPredicate::UGE;
}

static bool isFloatLePredicate(arith::CmpFPredicate predicate) {
  return predicate == arith::CmpFPredicate::OLE ||
         predicate == arith::CmpFPredicate::ULE;
}

static bool isReluPredicate(arith::CmpFPredicate predicate) {
  return predicate == arith::CmpFPredicate::OGT ||
         predicate == arith::CmpFPredicate::UGT;
}

static bool isReluPredicate(arith::CmpIPredicate predicate) {
  return predicate == arith::CmpIPredicate::sgt ||
         predicate == arith::CmpIPredicate::ugt;
}

static bool isZeroConstant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;

  Attribute attr = constant.getValue();
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
    if (!denseAttr.isSplat())
      return false;
    Attribute splat = denseAttr.getSplatValue<Attribute>();
    if (auto intAttr = dyn_cast<IntegerAttr>(splat))
      return intAttr.getValue().isZero();
    if (auto floatAttr = dyn_cast<FloatAttr>(splat))
      return floatAttr.getValue().isZero();
    return false;
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().isZero();
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValue().isZero();
  return false;
}

// ============================================================================
// PIM rewrite patterns
// ============================================================================

// Convert supported binary vector arithmetic/comparison rows to LLVMPIM rowops.
struct BinaryRowPattern : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    Operation *defOp = writeOp.getVector().getDefiningOp();
    if (!defOp)
      return failure();

    RowBinaryKind kind;
    Value lhs;
    Value rhs;

    if (auto op = dyn_cast<arith::AddFOp>(defOp)) {
      if (op.getLhs().getDefiningOp<arith::MulFOp>() ||
          op.getRhs().getDefiningOp<arith::MulFOp>())
        return failure();
      kind = RowBinaryKind::Add;
      lhs = op.getLhs();
      rhs = op.getRhs();
    } else if (auto op = dyn_cast<arith::AddIOp>(defOp)) {
      kind = RowBinaryKind::Add;
      lhs = op.getLhs();
      rhs = op.getRhs();
    } else if (auto op = dyn_cast<arith::MulFOp>(defOp)) {
      kind = RowBinaryKind::Mult;
      lhs = op.getLhs();
      rhs = op.getRhs();
    } else if (auto op = dyn_cast<arith::MulIOp>(defOp)) {
      kind = RowBinaryKind::Mult;
      lhs = op.getLhs();
      rhs = op.getRhs();
    } else if (auto op = dyn_cast<arith::SubFOp>(defOp)) {
      kind = RowBinaryKind::Sub;
      lhs = op.getLhs();
      rhs = op.getRhs();
    } else if (auto op = dyn_cast<arith::SubIOp>(defOp)) {
      kind = RowBinaryKind::Sub;
      lhs = op.getLhs();
      rhs = op.getRhs();
    } else if (auto op = dyn_cast<arith::DivFOp>(defOp)) {
      kind = RowBinaryKind::Div;
      lhs = op.getLhs();
      rhs = op.getRhs();
    } else if (auto op = dyn_cast<arith::DivSIOp>(defOp)) {
      kind = RowBinaryKind::Div;
      lhs = op.getLhs();
      rhs = op.getRhs();
    } else if (auto op = dyn_cast<arith::DivUIOp>(defOp)) {
      kind = RowBinaryKind::Div;
      lhs = op.getLhs();
      rhs = op.getRhs();
    } else if (auto op = dyn_cast<arith::CmpFOp>(defOp)) {
      if (isFloatGePredicate(op.getPredicate()))
        kind = RowBinaryKind::Ge;
      else if (isFloatLePredicate(op.getPredicate()))
        kind = RowBinaryKind::Le;
      else
        return failure();
      lhs = op.getLhs();
      rhs = op.getRhs();
    } else if (auto op = dyn_cast<arith::CmpIOp>(defOp)) {
      if (isIntegerGePredicate(op.getPredicate()))
        kind = RowBinaryKind::Ge;
      else if (isIntegerLePredicate(op.getPredicate()))
        kind = RowBinaryKind::Le;
      else
        return failure();
      lhs = op.getLhs();
      rhs = op.getRhs();
    } else {
      return failure();
    }

    auto lhsRead = lhs.getDefiningOp<vector::TransferReadOp>();
    auto rhsRead = rhs.getDefiningOp<vector::TransferReadOp>();
    if (!lhsRead || !rhsRead)
      return failure();
    if (!isSupportedVector(lhsRead.getResult()))
      return failure();
    if (!haveSameMemRefType(lhsRead.getBase(), rhsRead.getBase(),
                            writeOp.getBase()))
      return failure();

    std::optional<RowDataType> dataType =
        getRowDataTypeFromMemRef(lhsRead.getBase());
    if (!dataType)
      return failure();

    if (failed(createRowBinaryOp(rewriter, writeOp.getLoc(), kind, *dataType,
                                 lhsRead.getBase(), rhsRead.getBase(),
                                 writeOp.getBase())))
      return failure();

    rewriter.eraseOp(writeOp);
    llvm::errs() << "PIMPass: Replace binary vector op -> llvmpim.rowop_"
                 << getRowBinaryName(kind) << "\n";
    return success();
  }
};

// Convert supported unary vector math rows and ReLU selects to LLVMPIM rowops.
struct UnaryRowPattern : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    Operation *defOp = writeOp.getVector().getDefiningOp();
    if (!defOp)
      return failure();

    RowUnaryKind kind;
    Value input;

    if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
      Operation *conditionOp = selectOp.getCondition().getDefiningOp();
      if (auto cmpOp = dyn_cast_or_null<arith::CmpFOp>(conditionOp)) {
        if (!isReluPredicate(cmpOp.getPredicate()))
          return failure();
      } else if (auto cmpOp = dyn_cast_or_null<arith::CmpIOp>(conditionOp)) {
        if (!isReluPredicate(cmpOp.getPredicate()))
          return failure();
      } else {
        return failure();
      }

      auto inRead = selectOp.getTrueValue()
                        .getDefiningOp<vector::TransferReadOp>();
      if (!inRead || !isZeroConstant(selectOp.getFalseValue()))
        return failure();
      kind = RowUnaryKind::Relu;
      input = inRead.getResult();
    } else if (auto op = dyn_cast<math::SinOp>(defOp)) {
      kind = RowUnaryKind::Sin;
      input = op.getOperand();
    } else if (auto op = dyn_cast<math::CosOp>(defOp)) {
      kind = RowUnaryKind::Cos;
      input = op.getOperand();
    } else if (auto op = dyn_cast<math::AtanOp>(defOp)) {
      kind = RowUnaryKind::Atan;
      input = op.getOperand();
    } else if (auto op = dyn_cast<math::SinhOp>(defOp)) {
      kind = RowUnaryKind::Sinh;
      input = op.getOperand();
    } else if (auto op = dyn_cast<math::CoshOp>(defOp)) {
      kind = RowUnaryKind::Cosh;
      input = op.getOperand();
    } else if (auto op = dyn_cast<math::AtanhOp>(defOp)) {
      kind = RowUnaryKind::Atanh;
      input = op.getOperand();
    } else if (auto op = dyn_cast<math::SqrtOp>(defOp)) {
      kind = RowUnaryKind::Sqrt;
      input = op.getOperand();
    } else {
      return failure();
    }

    auto inputRead = input.getDefiningOp<vector::TransferReadOp>();
    if (!inputRead)
      return failure();
    if (!isSupportedVector(inputRead.getResult()))
      return failure();
    if (!haveSameMemRefType(inputRead.getBase(), writeOp.getBase()))
      return failure();

    std::optional<RowDataType> dataType =
        getRowDataTypeFromMemRef(inputRead.getBase());
    if (!dataType)
      return failure();

    if (failed(createRowUnaryOp(rewriter, writeOp.getLoc(), kind, *dataType,
                                inputRead.getBase(), writeOp.getBase())))
      return failure();

    rewriter.eraseOp(writeOp);
    llvm::errs() << "PIMPass: Replace unary vector op -> llvmpim.rowop_"
                 << getRowUnaryName(kind) << "\n";
    return success();
  }
};

// Convert linalg.softmax on memrefs to an LLVMPIM row softmax operation.
struct SoftmaxPattern : public OpRewritePattern<linalg::SoftmaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::SoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    Value output = op.getOutput();
    if (!haveSameMemRefType(input, output))
      return failure();

    std::optional<RowDataType> dataType = getRowDataTypeFromMemRef(input);
    if (!dataType)
      return failure();

    if (failed(createRowUnaryOp(rewriter, op.getLoc(), RowUnaryKind::Softmax,
                                *dataType, input, output)))
      return failure();

    rewriter.eraseOp(op);
    llvm::errs() << "PIMPass: Replace linalg.softmax -> llvmpim.rowop_softmax\n";
    return success();
  }
};

// Preserve the existing maxpool lowering for shifted row reads.
struct MaxPoolPattern : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    auto maxOp = writeOp.getVector().getDefiningOp<arith::MaximumFOp>();
    if (!maxOp)
      return failure();

    auto lhsRead = maxOp.getLhs().getDefiningOp<vector::TransferReadOp>();
    auto rhsRead = maxOp.getRhs().getDefiningOp<vector::TransferReadOp>();
    if (!lhsRead || !rhsRead)
      return failure();
    if (!isSupportedVector(lhsRead.getResult()))
      return failure();

    if (!haveSameMemRefType(lhsRead.getBase(), rhsRead.getBase(),
                            writeOp.getBase()))
      return failure();

    auto lhsIdx = lhsRead.getIndices();
    auto rhsIdx = rhsRead.getIndices();

    bool identical = true;
    for (auto pair : llvm::zip(lhsIdx, rhsIdx)) {
      if (std::get<0>(pair) != std::get<1>(pair)) {
        identical = false;
        break;
      }
    }
    if (identical)
      return failure();

    auto dataType = getRowDataTypeFromMemRef(lhsRead.getBase());
    if (!dataType || *dataType != RowDataType::I8)
      return failure();

    rewriter.create<llvmpim::RowOpMaxPool8>(
        writeOp.getLoc(), lhsRead.getBase(), rhsRead.getBase(),
        writeOp.getBase());
    rewriter.eraseOp(writeOp);

    llvm::errs() << "PIMPass: Replace maximumf -> llvmpim.rowop_maxpool8\n";
    return success();
  }
};

// ============================================================================
// Cleanup patterns for unfused vector transfers
// ============================================================================

// Replace unfused transfer_read with its base memref on the row path.
struct LowerUnfusedRead : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    // Keep reads that may still be consumed by row-op fusion patterns.
    for (OpOperand &use : op->getUses()) {
      Operation *user = use.getOwner();
      if (isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
              arith::SubFOp, arith::SubIOp, arith::DivFOp, arith::DivSIOp,
              arith::DivUIOp, arith::CmpFOp, arith::CmpIOp, arith::SelectOp,
              math::SinOp, math::CosOp, math::AtanOp, math::SinhOp,
              math::CoshOp, math::AtanhOp, math::SqrtOp>(user)) {
        return failure();
      }
    }

    rewriter.replaceOp(op, op.getBase());
    llvm::errs() << "PIMPass: Remove unfused transfer_read\n";
    return success();
  }
};

// Erase unfused transfer_write operations that are not committed by a row op.
struct LowerUnfusedWrite : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter &rewriter) const override {
    Operation *defOp = op.getVector().getDefiningOp();
    if (defOp &&
        isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
            arith::SubFOp, arith::SubIOp, arith::DivFOp, arith::DivSIOp,
            arith::DivUIOp, arith::CmpFOp, arith::CmpIOp, arith::SelectOp,
            arith::MaximumFOp, math::SinOp, math::CosOp, math::AtanOp,
            math::SinhOp, math::CoshOp, math::AtanhOp, math::SqrtOp>(defOp))
      return failure();

    rewriter.eraseOp(op);
    llvm::errs() << "PIMPass: Erase unfused transfer_write\n";
    return success();
  }
};

// ============================================================================
// Pass implementation
// ============================================================================

namespace {
struct ConvertVectorToLLVMPIMPass
    : public impl::ConvertVectorToLLVMPIMPassBase<
          ConvertVectorToLLVMPIMPass> {

  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<llvmpim::LLVMPIMDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void ConvertVectorToLLVMPIMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext &context = getContext();

  // --------------------------
  // Phase 1: PIM Rewriting
  // --------------------------
  {
    RewritePatternSet pim(&context);
    pim.add<BinaryRowPattern, UnaryRowPattern, SoftmaxPattern, MaxPoolPattern,
            LowerUnfusedRead, LowerUnfusedWrite>(&context);

    if (failed(applyPatternsAndFoldGreedily(module, std::move(pim)))) {
      signalPassFailure();
      return;
    }
  }

  // ---------------------------
  // Phase 2: Vector to LLVM
  // ---------------------------
  {
    RewritePatternSet patterns(&context);
    LLVMTypeConverter typeConverter(&context);

    populateVectorToLLVMConversionPatterns(typeConverter, patterns);

    ConversionTarget target(context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<llvmpim::LLVMPIMDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<math::MathDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<tensor::TensorDialect>();

    // The vector dialect must be fully converted by this phase.
    target.addIllegalDialect<vector::VectorDialect>();

    // Other operations are handled by the surrounding lowering pipeline.
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      (void)op;
      return true;
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
}

// ============================================================================
// Factory function
// ============================================================================

namespace mlir {
std::unique_ptr<Pass> createConvertVectorToLLVMPIMPass() {
  return std::make_unique<ConvertVectorToLLVMPIMPass>();
}
} // namespace mlir

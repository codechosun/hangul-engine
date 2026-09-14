// lib/Matrix.hpp
//
// 2차원 실수 배열. 행 우선(row-major)으로 한 덩어리에 담는다.
//
// B3 의 FVector 와 같은 설계다. 다른 것은 첨자가 둘이라는 것뿐이다.
// D1 의 FTensor 는 이걸 N차원으로 일반화한 것이 된다.

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "Types.h"
#include "Vector.hpp"

#include <cassert>
#include <cstddef>
#include <vector>

class FMatrix
{
public:
    FMatrix() = default;

    FMatrix(size_t InRows, size_t InCols)
        : Values(InRows * InCols, Real(0)), RowCount(InRows), ColCount(InCols)
    {
    }

    size_t Rows() const { return RowCount; }
    size_t Cols() const { return ColCount; }

    // 행 우선이므로 (Row, Col) 은 Row*ColCount + Col 이다.
    //
    // 이 한 줄을 클래스 안에 가두는 것이 요점이다. 밖에 흩어지면
    // 어딘가에서 Col*RowCount + Row 로 쓰게 되고, 그 버그는
    // 정사각 행렬에서는 안 드러난다.
    Real& operator()(size_t Row, size_t Col)
    {
        assert(Row < RowCount && Col < ColCount);
        return Values[Row * ColCount + Col];
    }

    const Real& operator()(size_t Row, size_t Col) const
    {
        assert(Row < RowCount && Col < ColCount);
        return Values[Row * ColCount + Col];
    }

    const Real* RowData(size_t Row) const
    {
        assert(Row < RowCount);
        return Values.data() + Row * ColCount;
    }

    Real* RowData(size_t Row)
    {
        assert(Row < RowCount);
        return Values.data() + Row * ColCount;
    }

    const Real* Data() const { return Values.data(); }
    Real* Data() { return Values.data(); }

    size_t Count() const { return Values.size(); }

    void Fill(Real Value);

private:
    std::vector<Real> Values;
    size_t RowCount = 0;
    size_t ColCount = 0;
};

// 행렬 곱하기 벡터. 결과의 길이는 행의 수와 같다.
FVector operator*(const FMatrix& M, const FVector& V);

// 전치한 것에 곱하는 것과 같다. 역전파에서 쓴다. (C2 예고)
FVector TransposedMultiply(const FMatrix& M, const FVector& V);

#endif

// lib/Matrix.cpp

#include "Matrix.hpp"

void FMatrix::Fill(Real Value)
{
    for (size_t i = 0; i < Values.size(); i++)
    {
        Values[i] = Value;
    }
}

FVector operator*(const FMatrix& M, const FVector& V)
{
    assert(M.Cols() == V.Size());

    FVector Result(M.Rows());

    for (size_t Row = 0; Row < M.Rows(); Row++)
    {
        const Real* Weights = M.RowData(Row);

        // 누적은 double 로. B3 에서와 같은 이유다.
        double Sum = 0.0;
        for (size_t Col = 0; Col < M.Cols(); Col++)
        {
            Sum += (double)Weights[Col] * (double)V[Col];
        }

        Result[Row] = (Real)Sum;
    }

    return Result;
}

FVector TransposedMultiply(const FMatrix& M, const FVector& V)
{
    assert(M.Rows() == V.Size());

    // 결과의 길이는 열의 수다.
    std::vector<double> Sums(M.Cols(), 0.0);

    // 행을 하나씩 훑는다. 전치 행렬을 실제로 만들지 않는다.
    //
    // 만들면 메모리도 쓰고 캐시도 어긋난다. 훑는 순서만 바꾸면
    // 원래 배치 그대로 순차 접근이 된다.
    for (size_t Row = 0; Row < M.Rows(); Row++)
    {
        const Real* Weights = M.RowData(Row);
        double Scale = (double)V[Row];

        for (size_t Col = 0; Col < M.Cols(); Col++)
        {
            Sums[Col] += (double)Weights[Col] * Scale;
        }
    }

    FVector Result(M.Cols());
    for (size_t Col = 0; Col < M.Cols(); Col++)
    {
        Result[Col] = (Real)Sums[Col];
    }

    return Result;
}

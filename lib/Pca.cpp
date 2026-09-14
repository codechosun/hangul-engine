// lib/Pca.cpp

#include "Pca.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace
{

// 대칭 행렬의 가장 큰 고유벡터를 찾는다.
//
// 아무 벡터에나 행렬을 계속 곱하면 가장 큰 고유값 쪽으로 쏠린다.
// 매번 길이를 1 로 되돌려주지 않으면 값이 넘치거나 사라진다.
std::vector<double> PowerIteration(const std::vector<double>& Covariance,
                                   size_t D, int Iterations, double& OutValue)
{
    std::vector<double> V(D, 0.0);

    // 시작 벡터. 0 이면 안 되고, 고유벡터와 직교하면 안 된다.
    // 규칙적인 값으로 채우면 두 조건 다 사실상 피해간다.
    for (size_t i = 0; i < D; i++)
    {
        V[i] = 1.0 / std::sqrt((double)D) * ((i % 2 == 0) ? 1.0 : -1.0);
    }

    std::vector<double> Next(D, 0.0);

    for (int Round = 0; Round < Iterations; Round++)
    {
        for (size_t i = 0; i < D; i++)
        {
            double Sum = 0.0;
            for (size_t j = 0; j < D; j++)
            {
                Sum += Covariance[i * D + j] * V[j];
            }
            Next[i] = Sum;
        }

        double Length = 0.0;
        for (size_t i = 0; i < D; i++)
        {
            Length += Next[i] * Next[i];
        }
        Length = std::sqrt(Length);

        if (Length < 1e-300)
        {
            break;
        }

        for (size_t i = 0; i < D; i++)
        {
            V[i] = Next[i] / Length;
        }
    }

    // 고유값 = v^T C v
    double Value = 0.0;
    for (size_t i = 0; i < D; i++)
    {
        double Sum = 0.0;
        for (size_t j = 0; j < D; j++)
        {
            Sum += Covariance[i * D + j] * V[j];
        }
        Value += V[i] * Sum;
    }

    OutValue = Value;

    // 부호를 못박는다.
    //
    // 고유벡터는 v 든 -v 든 똑같이 답이다. 그대로 두면 돌릴 때마다,
    // 언어마다 방향이 뒤집혀 비교를 못 한다. **절대값이 가장 큰 성분을
    // 양수로** 만들어 한 쪽으로 정한다. 파이썬 쪽도 같은 규칙을 쓴다.
    size_t Biggest = 0;
    for (size_t i = 1; i < D; i++)
    {
        if (std::fabs(V[i]) > std::fabs(V[Biggest]))
        {
            Biggest = i;
        }
    }

    if (V[Biggest] < 0.0)
    {
        for (size_t i = 0; i < D; i++)
        {
            V[i] = -V[i];
        }
    }

    return V;
}

} // namespace

FPca PcaFit(const FMatrix& Data, size_t ComponentCount, int Iterations)
{
    const size_t N = Data.Rows();
    const size_t D = Data.Cols();

    assert(N > 1);
    assert(ComponentCount <= D);

    FPca Result;
    Result.Mean = FVector(D);
    Result.Components = FMatrix(ComponentCount, D);
    Result.Variance = FVector(ComponentCount);

    // 1. 평균을 구해 뺀다.
    //
    // 안 빼면 "원점에서 얼마나 먼가"를 첫 주성분이 차지해버린다.
    // 우리가 알고 싶은 것은 **서로 얼마나 다른가**다.
    std::vector<double> Mean(D, 0.0);
    for (size_t n = 0; n < N; n++)
    {
        for (size_t d = 0; d < D; d++)
        {
            Mean[d] += (double)Data(n, d);
        }
    }
    for (size_t d = 0; d < D; d++)
    {
        Mean[d] /= (double)N;
        Result.Mean[d] = (Real)Mean[d];
    }

    // 2. 공분산 행렬. D x D 라 작다.
    std::vector<double> Covariance(D * D, 0.0);

    for (size_t n = 0; n < N; n++)
    {
        for (size_t i = 0; i < D; i++)
        {
            double Left = (double)Data(n, i) - Mean[i];
            for (size_t j = 0; j < D; j++)
            {
                Covariance[i * D + j] +=
                    Left * ((double)Data(n, j) - Mean[j]);
            }
        }
    }

    for (size_t i = 0; i < D * D; i++)
    {
        Covariance[i] /= (double)(N - 1);
    }

    double Total = 0.0;
    for (size_t i = 0; i < D; i++)
    {
        Total += Covariance[i * D + i];
    }
    Result.TotalVariance = (Real)Total;

    // 3. 가장 큰 방향부터 하나씩 뽑고, 뽑은 만큼 빼낸다(디플레이션).
    for (size_t c = 0; c < ComponentCount; c++)
    {
        double Value = 0.0;
        std::vector<double> V = PowerIteration(Covariance, D, Iterations, Value);

        for (size_t d = 0; d < D; d++)
        {
            Result.Components(c, d) = (Real)V[d];
        }
        Result.Variance[c] = (Real)Value;

        // C <- C - value * v v^T
        //
        // 방금 찾은 방향의 몫을 지운다. 그러면 다음 거듭제곱법이
        // 그 다음으로 큰 방향을 찾는다.
        for (size_t i = 0; i < D; i++)
        {
            for (size_t j = 0; j < D; j++)
            {
                Covariance[i * D + j] -= Value * V[i] * V[j];
            }
        }
    }

    return Result;
}

FMatrix PcaTransform(const FMatrix& Data, const FPca& Pca)
{
    const size_t N = Data.Rows();
    const size_t D = Data.Cols();
    const size_t K = Pca.Components.Rows();

    assert(D == Pca.Components.Cols());

    FMatrix Result(N, K);

    for (size_t n = 0; n < N; n++)
    {
        for (size_t k = 0; k < K; k++)
        {
            double Sum = 0.0;
            for (size_t d = 0; d < D; d++)
            {
                Sum += ((double)Data(n, d) - (double)Pca.Mean[d])
                     * (double)Pca.Components(k, d);
            }
            Result(n, k) = (Real)Sum;
        }
    }

    return Result;
}

// lib/GradCheck.cpp

#include "GradCheck.hpp"

#include <cmath>

double RelativeError(double A, double B)
{
    double Scale = std::fabs(A);
    if (std::fabs(B) > Scale)
    {
        Scale = std::fabs(B);
    }

    // 둘 다 아주 작으면 상대 오차를 말할 수 없다. 0 으로 친다.
    if (Scale < 1e-12)
    {
        return 0.0;
    }

    return std::fabs(A - B) / Scale;
}

FGradCheckResult GradCheck(const std::function<double()>& Loss,
                           Real* Parameters, const Real* Analytic,
                           int Count, double Step)
{
    FGradCheckResult Result;
    Result.Count = Count;

    for (int i = 0; i < Count; i++)
    {
        Real Saved = Parameters[i];

        // 중심 차분. 한쪽만 보는 것보다 오차가 훨씬 작다.
        // 이유는 C2 본문에 있다.
        Parameters[i] = (Real)((double)Saved + Step);
        double Plus = Loss();

        Parameters[i] = (Real)((double)Saved - Step);
        double Minus = Loss();

        Parameters[i] = Saved;   // 반드시 되돌려 놓는다

        double Numeric = (Plus - Minus) / (2.0 * Step);
        double Exact = (double)Analytic[i];

        double Relative = RelativeError(Numeric, Exact);
        double Absolute = std::fabs(Numeric - Exact);

        if (Relative > Result.WorstRelative)
        {
            Result.WorstRelative = Relative;
            Result.WorstAbsolute = Absolute;
            Result.WorstIndex = i;
            Result.WorstAnalytic = Exact;
            Result.WorstNumeric = Numeric;
        }
    }

    return Result;
}

FGradCheckResult GradCheckScattered(const std::function<double()>& Loss,
                                    Real* const* Parameters,
                                    const Real* const* Analytic,
                                    const int* Which, int Count, double Step)
{
    FGradCheckResult Result;
    Result.Count = Count;

    for (int n = 0; n < Count; n++)
    {
        const int i = (Which != nullptr) ? Which[n] : n;

        Real* Slot = Parameters[i];
        Real Saved = *Slot;

        *Slot = (Real)((double)Saved + Step);
        double Plus = Loss();

        *Slot = (Real)((double)Saved - Step);
        double Minus = Loss();

        *Slot = Saved;

        double Numeric = (Plus - Minus) / (2.0 * Step);
        double Exact = (double)(*Analytic[i]);

        double Relative = RelativeError(Numeric, Exact);
        double Absolute = std::fabs(Numeric - Exact);

        if (Relative > Result.WorstRelative)
        {
            Result.WorstRelative = Relative;
            Result.WorstAbsolute = Absolute;
            Result.WorstIndex = i;
            Result.WorstAnalytic = Exact;
            Result.WorstNumeric = Numeric;
        }
    }

    return Result;
}

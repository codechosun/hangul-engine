// lib/LogMath.c

#include "LogMath.h"

#include <math.h>

double LogAdd(double LogA, double LogB)
{
    // 한쪽이 -inf 면(확률 0) 다른 쪽이 답이다.
    if (LogA == -INFINITY) return LogB;
    if (LogB == -INFINITY) return LogA;

    double Big = (LogA > LogB) ? LogA : LogB;
    double Small = (LogA > LogB) ? LogB : LogA;

    // log(e^Big + e^Small) = Big + log(1 + e^(Small - Big))
    //
    // Small - Big 은 항상 0 이하이므로 e^(...) 가 넘치지 않는다.
    // log1p 를 쓰는 이유는 e^(...) 가 아주 작을 때 log(1 + x) 의
    // 유효숫자가 날아가기 때문이다. log1p 는 그 경우를 위해 있는 함수다.
    return Big + log1p(exp(Small - Big));
}

double LogSumExp(const double* LogValues, int Count)
{
    if (LogValues == NULL || Count <= 0)
    {
        return -INFINITY;
    }

    double Big = -INFINITY;
    for (int i = 0; i < Count; i++)
    {
        if (LogValues[i] > Big)
        {
            Big = LogValues[i];
        }
    }

    if (Big == -INFINITY)
    {
        return -INFINITY;
    }

    double Sum = 0.0;
    for (int i = 0; i < Count; i++)
    {
        Sum += exp(LogValues[i] - Big);
    }

    return Big + log(Sum);
}

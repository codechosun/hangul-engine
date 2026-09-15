// lib/Optimizer.cpp

#include "Optimizer.hpp"

#include <cmath>

void FAdamW::Init(size_t Count, const FAdamWConfig& InConfig)
{
    Config = InConfig;
    Moment.assign(Count, 0.0);
    Velocity.assign(Count, 0.0);
    Steps = 0;
}

void FAdamW::Reset()
{
    for (size_t i = 0; i < Moment.size(); i++)
    {
        Moment[i] = 0.0;
        Velocity[i] = 0.0;
    }
    Steps = 0;
}

size_t FAdamW::Bytes() const
{
    return (Moment.size() + Velocity.size()) * sizeof(double);
}

void FAdamW::Step(Real* const* Parameters, const Real* const* Gradients,
                  size_t Count, double Rate)
{
    Steps++;

    const double Use = (Rate >= 0.0) ? Rate : Config.Rate;

    // 편향 보정. m 과 v 를 0 에서 시작했으므로 초반에는 실제보다 작다.
    // 얼마나 작은지가 정확히 계산되므로 나눠준다.
    const double Correct1 = 1.0 - std::pow(Config.Beta1, (double)Steps);
    const double Correct2 = 1.0 - std::pow(Config.Beta2, (double)Steps);

    for (size_t i = 0; i < Count; i++)
    {
        const double G = (double)(*Gradients[i]);

        Moment[i] = Config.Beta1 * Moment[i] + (1.0 - Config.Beta1) * G;
        Velocity[i] = Config.Beta2 * Velocity[i] + (1.0 - Config.Beta2) * G * G;

        const double M = Moment[i] / Correct1;
        const double V = Velocity[i] / Correct2;

        double W = (double)(*Parameters[i]);

        W -= Use * M / (std::sqrt(V) + Config.Epsilon);

        // 여기가 AdamW 의 W 다. 그래디언트와 **무관하게** 적용된다.
        if (Config.WeightDecay > 0.0)
        {
            W -= Use * Config.WeightDecay * (double)(*Parameters[i]);
        }

        *Parameters[i] = (Real)W;
    }
}

double ClipGradients(Real* const* Gradients, size_t Count, double Limit)
{
    // 노름은 double 로 센다. 제곱을 더하는 계산이라 float 로는 금방 뭉갠다.
    double Square = 0.0;
    for (size_t i = 0; i < Count; i++)
    {
        const double G = (double)(*Gradients[i]);
        Square += G * G;
    }

    const double Norm = std::sqrt(Square);

    if (Limit > 0.0 && Norm > Limit)
    {
        // 방향은 그대로, 크기만 Limit 로.
        const double Shrink = Limit / Norm;
        for (size_t i = 0; i < Count; i++)
        {
            *Gradients[i] = (Real)((double)(*Gradients[i]) * Shrink);
        }
    }

    return Norm;
}

double ScheduleRate(double Base, size_t Step, size_t Warmup, size_t Total,
                    double MinRatio)
{
    if (Warmup > 0 && Step < Warmup)
    {
        // 0 에서 Base 까지. Step 0 에서 0 이 되지 않도록 한 칸 밀어둔다.
        return Base * ((double)(Step + 1) / (double)Warmup);
    }

    if (Total <= Warmup || Step >= Total)
    {
        return Base * MinRatio;
    }

    const double Progress = (double)(Step - Warmup) / (double)(Total - Warmup);
    const double Cosine = 0.5 * (1.0 + std::cos(3.14159265358979323846 * Progress));

    return Base * (MinRatio + (1.0 - MinRatio) * Cosine);
}

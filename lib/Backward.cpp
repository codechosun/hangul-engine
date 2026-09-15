// lib/Backward.cpp

#include "Backward.hpp"
#include "Attention.hpp"

#include <cassert>
#include <cmath>

namespace
{

const Real GEpsilon = (Real)1e-6;

void AddInto(FTensor& Target, const FTensor& Source)
{
    assert(Target.Count() == Source.Count());
    for (size_t i = 0; i < Target.Count(); i++)
    {
        Target.At(i) += Source.At(i);
    }
}

} // namespace

// ---------------------------------------------------------------- softmax

FTensor SoftmaxBackward(const FTensor& Probabilities, const FTensor& Upstream)
{
    assert(Probabilities.Count() == Upstream.Count());

    const size_t Last = Probabilities.Size(Probabilities.Rank() - 1);
    const size_t Rows = Probabilities.Count() / Last;

    FTensor Result(Probabilities.GetShape());

    for (size_t r = 0; r < Rows; r++)
    {
        const Real* P = Probabilities.Data() + r * Last;
        const Real* G = Upstream.Data() + r * Last;
        Real* Out = Result.Data() + r * Last;

        // 야코비안을 통째로 만들 필요가 없다.
        //
        //   dS_i = sum_j P_i (delta_ij - P_j) G_j
        //        = P_i G_i - P_i sum_j P_j G_j
        //
        // 안쪽 합이 i 에 안 걸리므로 한 번만 구하면 된다.
        // 야코비안은 (N x N) 인데 계산은 O(N) 이다.
        double Dot = 0.0;
        for (size_t i = 0; i < Last; i++)
        {
            Dot += (double)P[i] * (double)G[i];
        }

        for (size_t i = 0; i < Last; i++)
        {
            Out[i] = (Real)((double)P[i] * ((double)G[i] - Dot));
        }
    }

    return Result;
}

// ---------------------------------------------------------------- SiLU

FTensor SiluBackward(const FTensor& X, const FTensor& Upstream)
{
    assert(X.Count() == Upstream.Count());

    FTensor Result(X.GetShape());

    for (size_t i = 0; i < X.Count(); i++)
    {
        const double S = (double)Sigmoid(X.At(i));
        const double D = S * (1.0 + (double)X.At(i) * (1.0 - S));

        Result.At(i) = (Real)((double)Upstream.At(i) * D);
    }

    return Result;
}

// ---------------------------------------------------------------- RMSNorm

FTensor RmsNormBackward(const FTensor& X, const FTensor& Gain, Real Epsilon,
                        const FTensor& Upstream, FTensor& GainGrad)
{
    const size_t Last = X.Size(X.Rank() - 1);
    const size_t Rows = X.Count() / Last;

    assert(Gain.Count() == Last);
    assert(GainGrad.Count() == Last);

    FTensor Result(X.GetShape());

    for (size_t r = 0; r < Rows; r++)
    {
        const Real* Row = X.Data() + r * Last;
        const Real* U = Upstream.Data() + r * Last;
        Real* Out = Result.Data() + r * Last;

        double SquareSum = 0.0;
        for (size_t i = 0; i < Last; i++)
        {
            SquareSum += (double)Row[i] * (double)Row[i];
        }

        const double Scale =
            1.0 / std::sqrt(SquareSum / (double)Last + (double)Epsilon);

        // sum_i( u_i * g_i * x_i ). 줄 전체가 얽히는 지점이다.
        double Coupled = 0.0;
        for (size_t i = 0; i < Last; i++)
        {
            Coupled += (double)U[i] * (double)Gain.At(i) * (double)Row[i];
        }

        const double Shrink = Scale * Scale * Scale / (double)Last;

        for (size_t i = 0; i < Last; i++)
        {
            Out[i] = (Real)((double)Gain.At(i) * Scale * (double)U[i]
                          - Shrink * (double)Row[i] * Coupled);

            // 이득은 줄마다 공유된다. 그래서 **더한다.**
            GainGrad.At(i) = (Real)((double)GainGrad.At(i)
                                  + (double)U[i] * (double)Row[i] * Scale);
        }
    }

    return Result;
}

// ---------------------------------------------------------------- FDense

void FDenseGrad::Init(const FDense& Source)
{
    Weight = FTensor(Source.Weight.GetShape());
    Bias = FTensor(Source.Bias.GetShape());
    Zero();
}

void FDenseGrad::Zero()
{
    Weight.Fill(Real(0));
    Bias.Fill(Real(0));
}

FTensor DenseBackward(const FDense& Dense, const FTensor& X,
                      const FTensor& Upstream, FDenseGrad& Grad)
{
    const size_t In = Dense.InSize();
    const size_t Out = Dense.OutSize();
    const size_t Rows = X.Count() / In;

    FTensor FlatX = X.Reshaped({ Rows, In });
    FTensor FlatU = Upstream.Reshaped({ Rows, Out });

    // dW = X^T U
    AddInto(Grad.Weight, MatMul(FlatX.Transposed(0, 1), FlatU));

    if (Dense.bUseBias)
    {
        for (size_t r = 0; r < Rows; r++)
        {
            for (size_t o = 0; o < Out; o++)
            {
                Grad.Bias.At(o) += FlatU(r, o);
            }
        }
    }

    // dX = U W^T
    FTensor FlatResult = MatMul(FlatU, Dense.Weight.Transposed(0, 1));

    return FlatResult.Reshaped(X.GetShape());
}

// ---------------------------------------------------------------- 어텐션

FAttentionGrad AttendBackward(const FTensor& Q, const FTensor& K,
                              const FTensor& V, const FTensor& Weights,
                              const FTensor& Upstream)
{
    assert(Q.Rank() == 4);

    const size_t Dim = Q.Size(3);
    const Real Scale = (Real)(1.0 / std::sqrt((double)Dim));

    // O = P V
    FAttentionGrad Result;
    Result.V = MatMul(Weights.Transposed(2, 3), Upstream);

    FTensor UpstreamP = MatMul(Upstream, V.Transposed(2, 3));

    // softmax 를 거슬러 올라간다.
    //
    // 마스크 처리를 여기서 안 하는 이유. 막힌 자리는 P 가 0 이므로
    // dS = P (G - dot) 도 0 이다. 순전파에서 눌러둔 것이 역전파에서
    // 저절로 지워진다. **-1e30 을 쓴 값이 여기서도 돌아온다.**
    FTensor UpstreamScores = SoftmaxBackward(Weights, UpstreamP);

    // S = Q K^T * Scale
    Result.Q = MatMul(UpstreamScores, K) * Scale;
    Result.K = MatMul(UpstreamScores.Transposed(2, 3), Q) * Scale;

    return Result;
}

// ---------------------------------------------------------------- 블록

void FBlockGrad::Init(const FBlock& Source)
{
    Query.Init(Source.Query);
    Key.Init(Source.Key);
    Value.Init(Source.Value);
    Project.Init(Source.Project);
    Up.Init(Source.Up);
    Down.Init(Source.Down);

    AttentionGain = FTensor(Source.AttentionGain.GetShape());
    FeedGain = FTensor(Source.FeedGain.GetShape());

    Zero();
}

void FBlockGrad::Zero()
{
    Query.Zero();
    Key.Zero();
    Value.Zero();
    Project.Zero();
    Up.Zero();
    Down.Zero();
    AttentionGain.Fill(Real(0));
    FeedGain.Fill(Real(0));
}

FTensor BlockForwardTrace(const FBlock& Block, const FTensor& X,
                          FBlockTrace& Trace)
{
    const FBlockConfig& Config = Block.GetConfig();

    // 이 판이 다루는 설정. D3 의 스위치는 여기서 고정한다.
    assert(Config.bNorm && Config.bPreNorm && Config.bResidual);
    assert(!Config.bQkNorm);

    Trace.Input = X;
    Trace.Normed = RmsNorm(X, Block.AttentionGain, GEpsilon);

    Trace.Q = SplitHeads(Block.Query.Forward(Trace.Normed), Config.Heads);
    Trace.K = SplitHeads(Block.Key.Forward(Trace.Normed), Config.Heads);
    Trace.V = SplitHeads(Block.Value.Forward(Trace.Normed), Config.Heads);

    FAttention Attended = Attend(Trace.Q, Trace.K, Trace.V, Config.bCausal);
    Trace.Weights = std::move(Attended.Weights);
    Trace.Attended = std::move(Attended.Output);

    Trace.Merged = MergeHeads(Trace.Attended);
    Trace.AfterAttention = Block.Project.Forward(Trace.Merged) + X;

    Trace.FeedNormed = RmsNorm(Trace.AfterAttention, Block.FeedGain, GEpsilon);
    Trace.UpOut = Block.Up.Forward(Trace.FeedNormed);
    Trace.Activated = Silu(Trace.UpOut);

    return Block.Down.Forward(Trace.Activated) + Trace.AfterAttention;
}

FTensor BlockBackward(const FBlock& Block, const FBlockTrace& Trace,
                      const FTensor& Upstream, FBlockGrad& Grad)
{
    const FBlockConfig& Config = Block.GetConfig();

    // ---- 앞먹임 쪽 ----
    //
    // Result = Down(Silu(Up(Norm(A)))) + A
    //
    // A 가 **두 길로 갈라진다.** 한 번은 앞먹임을 타고, 한 번은 곧장.
    // 그래서 dA 는 두 곳에서 와서 더해진다.
    FTensor Down1 = DenseBackward(Block.Down, Trace.Activated, Upstream,
                                  Grad.Down);
    FTensor Up1 = SiluBackward(Trace.UpOut, Down1);
    FTensor Normed1 = DenseBackward(Block.Up, Trace.FeedNormed, Up1, Grad.Up);

    FTensor AfterAttentionGrad = RmsNormBackward(
        Trace.AfterAttention, Block.FeedGain, GEpsilon, Normed1,
        Grad.FeedGain);

    // 잔차. 지름길로 온 몫을 더한다.
    AddInto(AfterAttentionGrad, Upstream);

    // ---- 어텐션 쪽 ----
    //
    // AfterAttention = Project(Merge(Attend(...))) + X
    FTensor MergedGrad = DenseBackward(Block.Project, Trace.Merged,
                                       AfterAttentionGrad, Grad.Project);

    // MergeHeads 의 역은 SplitHeads 다. 축을 도로 바꿔놓는 것뿐이다.
    FTensor AttendedGrad = SplitHeads(MergedGrad, Config.Heads);

    FAttentionGrad Qkv = AttendBackward(Trace.Q, Trace.K, Trace.V,
                                        Trace.Weights, AttendedGrad);

    // Q, K, V 가 **같은 Normed 에서 나왔다.** 셋 다 더해야 한다.
    FTensor NormedGrad = DenseBackward(Block.Query, Trace.Normed,
                                       MergeHeads(Qkv.Q), Grad.Query);
    AddInto(NormedGrad, DenseBackward(Block.Key, Trace.Normed,
                                      MergeHeads(Qkv.K), Grad.Key));
    AddInto(NormedGrad, DenseBackward(Block.Value, Trace.Normed,
                                      MergeHeads(Qkv.V), Grad.Value));

    FTensor InputGrad = RmsNormBackward(Trace.Input, Block.AttentionGain,
                                        GEpsilon, NormedGrad,
                                        Grad.AttentionGain);

    // 여기도 잔차.
    AddInto(InputGrad, AfterAttentionGrad);

    return InputGrad;
}

// ---------------------------------------------------------------- 모델

void FTransformerGrad::Init(const FTransformer& Source)
{
    TokenEmbedding = FTensor(Source.TokenEmbedding.GetShape());
    PositionEmbedding = FTensor(Source.PositionEmbedding.GetShape());
    FinalGain = FTensor(Source.FinalGain.GetShape());
    Head.Init(Source.Head);

    Blocks.clear();
    for (size_t i = 0; i < Source.Blocks.size(); i++)
    {
        FBlockGrad One;
        One.Init(Source.Blocks[i]);
        Blocks.push_back(std::move(One));
    }

    Zero();
}

void FTransformerGrad::Zero()
{
    TokenEmbedding.Fill(Real(0));
    PositionEmbedding.Fill(Real(0));
    FinalGain.Fill(Real(0));
    Head.Zero();

    for (size_t i = 0; i < Blocks.size(); i++)
    {
        Blocks[i].Zero();
    }
}

FTensor TransformerForwardTrace(const FTransformer& Model,
                                const uint32_t* Tokens, size_t Batch,
                                size_t Length, FModelTrace& Trace)
{
    const FModelConfig& Config = Model.Config;

    Trace.Batch = Batch;
    Trace.Length = Length;
    Trace.Tokens.assign(Tokens, Tokens + Batch * Length);

    FTensor X({ Batch, Length, Config.Model });

    for (size_t b = 0; b < Batch; b++)
    {
        for (size_t t = 0; t < Length; t++)
        {
            const uint32_t Token = Tokens[b * Length + t];
            assert((size_t)Token < Config.Vocab);

            for (size_t d = 0; d < Config.Model; d++)
            {
                X(b, t, d) = Model.TokenEmbedding((size_t)Token, d)
                           + Model.PositionEmbedding(t, d);
            }
        }
    }

    Trace.Blocks.clear();
    Trace.Blocks.resize(Model.Blocks.size());

    for (size_t i = 0; i < Model.Blocks.size(); i++)
    {
        X = BlockForwardTrace(Model.Blocks[i], X, Trace.Blocks[i]);
    }

    Trace.Final = X;
    Trace.Normed = RmsNorm(Trace.Final, Model.FinalGain, GEpsilon);

    return Model.Head.Forward(Trace.Normed);
}

void TransformerBackward(const FTransformer& Model, const FModelTrace& Trace,
                         const FTensor& UpstreamLogits, FTransformerGrad& Grad)
{
    const FModelConfig& Config = Model.Config;

    FTensor NormedGrad = DenseBackward(Model.Head, Trace.Normed,
                                       UpstreamLogits, Grad.Head);

    FTensor X = RmsNormBackward(Trace.Final, Model.FinalGain, GEpsilon,
                                NormedGrad, Grad.FinalGain);

    // 블록을 **거꾸로** 훑는다.
    for (size_t i = Model.Blocks.size(); i-- > 0; )
    {
        X = BlockBackward(Model.Blocks[i], Trace.Blocks[i], X, Grad.Blocks[i]);
    }

    // 임베딩. 여기가 가중치 공유가 가장 노골적으로 드러나는 자리다.
    //
    // 같은 토큰이 열 자리에 나오면 그 줄의 그래디언트는 **열 번 더해진다.**
    // 위치 임베딩은 배치 전체가 같은 줄을 쓰므로 배치 수만큼 더해진다.
    //
    // 덮어쓰면 마지막 자리 것만 남아서 조용히 틀린다.
    for (size_t b = 0; b < Trace.Batch; b++)
    {
        for (size_t t = 0; t < Trace.Length; t++)
        {
            const uint32_t Token = Trace.Tokens[b * Trace.Length + t];

            for (size_t d = 0; d < Config.Model; d++)
            {
                const Real G = X(b, t, d);

                Grad.TokenEmbedding((size_t)Token, d) += G;
                Grad.PositionEmbedding(t, d) += G;
            }
        }
    }
}

// ---------------------------------------------------------------- 손실

double CrossEntropyLoss(const FTensor& Logits, const uint32_t* Targets)
{
    assert(Logits.Rank() == 3);

    const size_t Batch = Logits.Size(0);
    const size_t Length = Logits.Size(1);
    const size_t Vocab = Logits.Size(2);

    double Total = 0.0;

    for (size_t r = 0; r < Batch * Length; r++)
    {
        const Real* Row = Logits.Data() + r * Vocab;

        double Biggest = (double)Row[0];
        for (size_t v = 1; v < Vocab; v++)
        {
            if ((double)Row[v] > Biggest) Biggest = (double)Row[v];
        }

        double Sum = 0.0;
        for (size_t v = 0; v < Vocab; v++)
        {
            Sum += std::exp((double)Row[v] - Biggest);
        }

        // log p = (x - max) - log(sum exp(x - max))
        const size_t Target = (size_t)Targets[r];
        Total -= ((double)Row[Target] - Biggest) - std::log(Sum);
    }

    return Total / (double)(Batch * Length);
}

FTensor CrossEntropyBackward(const FTensor& Logits, const uint32_t* Targets)
{
    const size_t Batch = Logits.Size(0);
    const size_t Length = Logits.Size(1);
    const size_t Vocab = Logits.Size(2);
    const double Count = (double)(Batch * Length);

    FTensor Result(Logits.GetShape());

    for (size_t r = 0; r < Batch * Length; r++)
    {
        const Real* Row = Logits.Data() + r * Vocab;
        Real* Out = Result.Data() + r * Vocab;

        double Biggest = (double)Row[0];
        for (size_t v = 1; v < Vocab; v++)
        {
            if ((double)Row[v] > Biggest) Biggest = (double)Row[v];
        }

        double Sum = 0.0;
        for (size_t v = 0; v < Vocab; v++)
        {
            Sum += std::exp((double)Row[v] - Biggest);
        }

        // (p - y) / N. C2 에서 유도한 그대로다.
        for (size_t v = 0; v < Vocab; v++)
        {
            double P = std::exp((double)Row[v] - Biggest) / Sum;
            Out[v] = (Real)((P - ((size_t)Targets[r] == v ? 1.0 : 0.0)) / Count);
        }
    }

    return Result;
}

// ---------------------------------------------------------------- 배선

namespace
{

void PushTensor(FTensor& T, std::vector<Real*>& Out)
{
    for (size_t i = 0; i < T.Count(); i++)
    {
        Out.push_back(&T.At(i));
    }
}

} // namespace

void CollectParameters(FTransformer& Model, std::vector<Real*>& Out)
{
    Out.clear();

    PushTensor(Model.TokenEmbedding, Out);
    PushTensor(Model.PositionEmbedding, Out);

    for (size_t i = 0; i < Model.Blocks.size(); i++)
    {
        FBlock& B = Model.Blocks[i];
        PushTensor(B.Query.Weight, Out);
        PushTensor(B.Key.Weight, Out);
        PushTensor(B.Value.Weight, Out);
        PushTensor(B.Project.Weight, Out);
        PushTensor(B.Up.Weight, Out);
        PushTensor(B.Down.Weight, Out);
        PushTensor(B.AttentionGain, Out);
        PushTensor(B.FeedGain, Out);
    }

    PushTensor(Model.FinalGain, Out);
    PushTensor(Model.Head.Weight, Out);
}

void CollectGradients(FTransformerGrad& Grad, std::vector<Real*>& Out)
{
    Out.clear();

    PushTensor(Grad.TokenEmbedding, Out);
    PushTensor(Grad.PositionEmbedding, Out);

    for (size_t i = 0; i < Grad.Blocks.size(); i++)
    {
        FBlockGrad& B = Grad.Blocks[i];
        PushTensor(B.Query.Weight, Out);
        PushTensor(B.Key.Weight, Out);
        PushTensor(B.Value.Weight, Out);
        PushTensor(B.Project.Weight, Out);
        PushTensor(B.Up.Weight, Out);
        PushTensor(B.Down.Weight, Out);
        PushTensor(B.AttentionGain, Out);
        PushTensor(B.FeedGain, Out);
    }

    PushTensor(Grad.FinalGain, Out);
    PushTensor(Grad.Head.Weight, Out);
}

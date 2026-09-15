// lib/Model.cpp

#include "Model.hpp"
#include "Attention.hpp"

#include <cassert>
#include <cmath>

FKvCache::FKvCache(const FModelConfig& Config)
{
    const size_t Head = Config.Model / Config.Heads;

    for (size_t i = 0; i < Config.Layers; i++)
    {
        Keys.push_back(FTensor({ 1, Config.Heads, Config.MaxLength, Head }));
        Values.push_back(FTensor({ 1, Config.Heads, Config.MaxLength, Head }));
    }

    Length = 0;
}

void FKvCache::Clear()
{
    for (size_t i = 0; i < Keys.size(); i++)
    {
        Keys[i].Fill(Real(0));
        Values[i].Fill(Real(0));
    }
    Length = 0;
}

size_t FKvCache::Bytes() const
{
    size_t Total = 0;
    for (size_t i = 0; i < Keys.size(); i++)
    {
        Total += Keys[i].Count() * sizeof(Real);
        Total += Values[i].Count() * sizeof(Real);
    }
    return Total;
}

FTransformer::FTransformer(const FModelConfig& InConfig)
    : Config(InConfig),
      TokenEmbedding({ InConfig.Vocab, InConfig.Model }),
      PositionEmbedding({ InConfig.MaxLength, InConfig.Model }),
      FinalGain({ InConfig.Model }),
      Head(InConfig.Model, InConfig.Vocab, false)
{
    FinalGain.Fill(Real(1));

    FBlockConfig BlockConfig;
    BlockConfig.Model = InConfig.Model;
    BlockConfig.Heads = InConfig.Heads;
    BlockConfig.Hidden = InConfig.Hidden;

    for (size_t i = 0; i < InConfig.Layers; i++)
    {
        Blocks.push_back(FBlock(BlockConfig));
    }
}

void FTransformer::Init(FRandom& Rng)
{
    const double Range = 1.0 / std::sqrt((double)Config.Model);

    for (size_t i = 0; i < TokenEmbedding.Count(); i++)
    {
        TokenEmbedding.At(i) = (Real)RandomRange(&Rng, Range);
    }
    for (size_t i = 0; i < PositionEmbedding.Count(); i++)
    {
        PositionEmbedding.At(i) = (Real)RandomRange(&Rng, Range);
    }

    for (size_t i = 0; i < Blocks.size(); i++)
    {
        Blocks[i].Init(Rng);
    }

    Head.Init(Rng);
}

size_t FTransformer::ParameterCount() const
{
    size_t Total = TokenEmbedding.Count() + PositionEmbedding.Count()
                 + FinalGain.Count() + Head.Weight.Count();

    for (size_t i = 0; i < Blocks.size(); i++)
    {
        const FBlock& B = Blocks[i];
        Total += B.Query.Weight.Count() + B.Key.Weight.Count()
               + B.Value.Weight.Count() + B.Project.Weight.Count()
               + B.Up.Weight.Count() + B.Down.Weight.Count()
               + B.AttentionGain.Count() + B.FeedGain.Count();
    }

    return Total;
}

FTensor FTransformer::Forward(const uint32_t* Tokens, size_t Batch,
                              size_t Length) const
{
    assert(Tokens != nullptr);
    assert(Length <= Config.MaxLength);

    // 1. 임베딩을 꺼내고 위치를 더한다.
    //
    // 위치를 안 더하면 어텐션은 순서를 모른다. 어텐션 자체가
    // 위치에 무관한 연산이기 때문이다. "가나다" 와 "다나가" 를 구별 못 한다.
    FTensor X({ Batch, Length, Config.Model });

    for (size_t b = 0; b < Batch; b++)
    {
        for (size_t t = 0; t < Length; t++)
        {
            const uint32_t Token = Tokens[b * Length + t];
            assert((size_t)Token < Config.Vocab);

            for (size_t d = 0; d < Config.Model; d++)
            {
                X(b, t, d) = TokenEmbedding((size_t)Token, d)
                           + PositionEmbedding(t, d);
            }
        }
    }

    // 2. 블록을 통과시킨다.
    for (size_t i = 0; i < Blocks.size(); i++)
    {
        X = Blocks[i].Forward(X);
    }

    // 3. 마지막 정규화.
    //
    // Pre-Norm 은 블록 안쪽만 정규화하므로 마지막 출력이 정규화되지 않는다.
    // 그대로 출력층에 넣으면 점수가 커진다. (D3 연습문제 3)
    X = RmsNorm(X, FinalGain, (Real)1e-6);

    // 4. 어휘 크기의 점수로.
    return Head.Forward(X);
}

FTensor FTransformer::Step(uint32_t Token, FKvCache& Cache) const
{
    assert((size_t)Token < Config.Vocab);
    assert(Cache.Length < Config.MaxLength);
    assert(Cache.Keys.size() == Config.Layers);

    const size_t Position = Cache.Length;
    const size_t HeadDim = Config.Model / Config.Heads;

    // 토큰 하나짜리 입력. (1, 1, 모델차원)
    FTensor X({ 1, 1, Config.Model });
    for (size_t d = 0; d < Config.Model; d++)
    {
        X(0, 0, d) = TokenEmbedding((size_t)Token, d)
                   + PositionEmbedding(Position, d);
    }

    for (size_t Layer = 0; Layer < Blocks.size(); Layer++)
    {
        const FBlock& B = Blocks[Layer];

        FTensor Normed = RmsNorm(X, B.AttentionGain, (Real)1e-6);

        FTensor Q = SplitHeads(B.Query.Forward(Normed), Config.Heads);
        FTensor K = SplitHeads(B.Key.Forward(Normed), Config.Heads);
        FTensor V = SplitHeads(B.Value.Forward(Normed), Config.Heads);

        // 이번 자리의 K, V 를 캐시에 적어둔다.
        for (size_t h = 0; h < Config.Heads; h++)
        {
            for (size_t d = 0; d < HeadDim; d++)
            {
                Cache.Keys[Layer](0, h, Position, d) = K(0, h, 0, d);
                Cache.Values[Layer](0, h, Position, d) = V(0, h, 0, d);
            }
        }

        // 지금까지의 K, V 만 잘라낸다.
        //
        // 여기서 복사가 한 번 일어난다. 진짜 구현은 캐시를 그대로 보게
        // 만들지만, 그러려면 텐서에 "부분만 보기"가 있어야 한다. (D1 연습문제 2)
        const size_t Visible = Position + 1;

        FTensor UsedK({ 1, Config.Heads, Visible, HeadDim });
        FTensor UsedV({ 1, Config.Heads, Visible, HeadDim });

        for (size_t h = 0; h < Config.Heads; h++)
        {
            for (size_t t = 0; t < Visible; t++)
            {
                for (size_t d = 0; d < HeadDim; d++)
                {
                    UsedK(0, h, t, d) = Cache.Keys[Layer](0, h, t, d);
                    UsedV(0, h, t, d) = Cache.Values[Layer](0, h, t, d);
                }
            }
        }

        // 마스크가 필요 없다. 볼 수 있는 것만 넘겼기 때문이다.
        FAttention Attended = Attend(Q, UsedK, UsedV, false);

        FTensor AttentionOut = B.Project.Forward(MergeHeads(Attended.Output));
        X = std::move(AttentionOut) + X;

        FTensor FeedNormed = RmsNorm(X, B.FeedGain, (Real)1e-6);
        FTensor FeedOut = B.Down.Forward(Silu(B.Up.Forward(FeedNormed)));
        X = std::move(FeedOut) + X;
    }

    Cache.Length++;

    X = RmsNorm(X, FinalGain, (Real)1e-6);
    FTensor Logits = Head.Forward(X);

    // (1, 1, 어휘) 를 (어휘) 로 눕힌다.
    return Logits.Reshaped({ Config.Vocab });
}

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

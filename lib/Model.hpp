// lib/Model.hpp
//
// 트랜스포머 한 채. 조각을 전부 이어 붙인다.
//
//     토큰 -> 임베딩 + 위치 -> 블록 x N -> 최종 정규화 -> 출력층 -> 점수
//
// 그리고 **글자를 하나씩 뽑는 길**을 따로 둔다.
//
// 왜 따로 두는가
//   생성할 때는 토큰을 하나 붙일 때마다 앞의 것들을 다시 계산할 필요가 없다.
//   이미 구한 K 와 V 는 안 바뀌기 때문이다. 들고 있으면 매번 한 칸만 하면 된다.
//   그게 KV 캐시이고, 실제 추론 서버 메모리의 대부분을 차지하는 것이기도 하다.

#ifndef MODEL_HPP
#define MODEL_HPP

#include "Block.hpp"
#include "Random.h"
#include "Tensor.hpp"
#include "Types.h"

#include <cstdint>
#include <vector>

struct FModelConfig
{
    size_t Vocab = 64;
    size_t Model = 64;
    size_t Heads = 4;
    size_t Hidden = 256;
    size_t Layers = 2;
    size_t MaxLength = 64;
};

// 층마다 지금까지의 K 와 V 를 담아둔다.
struct FKvCache
{
    FKvCache() = default;
    FKvCache(const FModelConfig& Config);

    void Clear();

    std::vector<FTensor> Keys;     // 층마다 (1, 헤드, 최대길이, 헤드차원)
    std::vector<FTensor> Values;
    size_t Length = 0;             // 지금까지 담은 토큰 수

    size_t Bytes() const;
};

class FTransformer
{
public:
    FTransformer() = default;
    explicit FTransformer(const FModelConfig& InConfig);

    void Init(FRandom& Rng);

    size_t ParameterCount() const;

    // 통째로 한 번에. Tokens 는 (배치 x 길이) 개.
    // 결과는 (배치, 길이, 어휘) 점수다.
    FTensor Forward(const uint32_t* Tokens, size_t Batch, size_t Length) const;

    // 토큰 하나를 밀어넣고 그 자리의 점수만 받는다. 캐시를 갱신한다.
    // 결과는 (어휘) 짜리다.
    FTensor Step(uint32_t Token, FKvCache& Cache) const;

    FModelConfig Config;

    FTensor TokenEmbedding;      // (어휘, 모델차원)
    FTensor PositionEmbedding;   // (최대길이, 모델차원)
    std::vector<FBlock> Blocks;
    FTensor FinalGain;           // (모델차원)
    FDense Head;                 // (모델차원, 어휘)
};

#endif

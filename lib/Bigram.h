// lib/Bigram.h
//
// 앞 글자 하나를 보고 다음 글자를 예측하는 모델.
//
// 코퍼스의 한 줄을 한 문장으로 본다. 줄마다 앞뒤에 특수 토큰을 붙인다.
//
//     [BOS] 한 국 어 [EOS]
//
// BOS 는 "여기서 시작한다", EOS 는 "여기서 끝난다"를 뜻한다.
// 이 둘이 없으면 첫 글자를 무엇으로 뽑을지, 언제 멈출지 알 수가 없다.

#ifndef BIGRAM_H
#define BIGRAM_H

#include <stdint.h>

#include "Random.h"
#include "Utf8.h"

#ifdef __cplusplus
extern "C" {
#endif

// 특수 토큰은 유니코드 밖의 번호를 쓴다.
// 코드포인트는 0x10FFFF 까지이므로 그 위는 영원히 빈다.
// 진짜 글자와 절대 부딪히지 않는다.
#define TOKEN_BOS   (CODEPOINT_LIMIT)        // 0x110000
#define TOKEN_EOS   (CODEPOINT_LIMIT + 1u)   // 0x110001
#define VOCAB_LIMIT (CODEPOINT_LIMIT + 2u)   // 0x110002

// 압축 행 저장(CSR). 앞 글자별로 "다음에 온 글자들"을 한 줄로 붙여 담는다.
//
//   Start[Prev] ~ Start[Prev+1] 이 앞 글자 Prev 의 구간이고,
//   그 구간 안에서 Next[i] 가 다음 글자, Cumulative[i] 가 구간 안의 누적 빈도다.
//
// 해시맵은 "이 쌍이 몇 번 나왔나"는 답하지만
// "이 글자 다음에 올 수 있는 것들을 전부 달라"에는 답하지 못한다.
// 뽑으려면 후보 목록이 한 자리에 모여 있어야 한다.
typedef struct
{
    uint32_t* Next;         // 길이 PairCount
    uint64_t* Cumulative;   // 길이 PairCount. 구간마다 0 에서 다시 쌓는다
    uint32_t* Start;        // 길이 VOCAB_LIMIT + 1

    uint64_t PairCount;     // 서로 다른 (앞, 뒤) 쌍의 개수
    uint64_t Total;         // 센 쌍의 총 개수
    uint64_t LineCount;     // 문장(줄)의 개수
} FBigram;

// 코퍼스를 읽어 모델을 만든다. 성공하면 1.
int BigramBuild(FBigram* Model, const char* Path);

void BigramFree(FBigram* Model);

// Prev 다음에 Next 가 온 횟수. 없으면 0.
uint64_t BigramCount(const FBigram* Model, uint32_t Prev, uint32_t Next);

// Prev 가 앞 글자로 나온 총 횟수. 없으면 0.
uint64_t BigramContextTotal(const FBigram* Model, uint32_t Prev);

// Prev 다음에 올 수 있는 글자의 가짓수.
uint32_t BigramChoiceCount(const FBigram* Model, uint32_t Prev);

// Prev 다음에 올 글자를 확률에 비례해 하나 뽑는다.
// 본 적 없는 앞 글자면 TOKEN_EOS.
uint32_t BigramPick(const FBigram* Model, uint32_t Prev, FRandom* Rng);

// 문장 하나를 만들어 Out 에 담는다. 담은 글자 수를 돌려준다.
// EOS 가 나오거나 MaxLength 에 닿으면 멈춘다. BOS/EOS 는 담지 않는다.
int BigramGenerate(const FBigram* Model, FRandom* Rng,
                   uint32_t* Out, int MaxLength);

#ifdef __cplusplus
}
#endif
#endif

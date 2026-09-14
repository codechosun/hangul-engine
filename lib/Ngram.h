// lib/Ngram.h
//
// 앞 N-1 글자를 보고 다음 글자를 예측하는 모델.
//
// A5 의 lib/Bigram 을 N 으로 일반화한 것이다. N=2 로 두면 같은 모델이 된다.
//
// 문장 하나는 이렇게 만든다. 앞을 BOS 로 N-1 개 채운다.
//
//     N=3, "한국" ->  [BOS][BOS] 한 국 [EOS]
//
// 그래야 첫 글자를 뽑을 때도 앞 문맥이 N-1 개 있다.
// 이 규칙에서는 **N 이 무엇이든 그램의 개수가 같다.**
//
//     그램 수 = 글자 수 + 문장 수

#ifndef NGRAM_H
#define NGRAM_H

#include <stdint.h>

#include "Bigram.h"   // TOKEN_BOS, TOKEN_EOS, VOCAB_LIMIT
#include "Random.h"

#ifdef __cplusplus
extern "C" {
#endif

// 한 그램에 담을 수 있는 최대 토큰 수.
#define NGRAM_MAX_ORDER 8

typedef struct
{
    int Order;              // N

    uint32_t* Grams;        // GramCount * Order 개. 사전순으로 정렬되어 있다
    uint64_t* Cumulative;   // 문맥 구간 안에서의 누적 빈도
    uint64_t  GramCount;

    // 같은 문맥을 가진 그램들이 몇 번부터 몇 번까지인가.
    uint64_t* ContextStart; // 길이 ContextCount + 1
    uint64_t  ContextCount;

    uint64_t  Total;        // 센 그램의 총 개수
    uint64_t  OnceCount;    // 딱 한 번 나온 그램의 개수
    uint64_t  LineCount;    // 문장 수
    uint64_t  CharCount;    // 줄바꿈을 뺀 글자 수
} FNgram;

// 코퍼스의 앞 MaxLines 줄로 모델을 만든다. MaxLines 가 0 이면 전부.
// 성공하면 1.
int NgramBuild(FNgram* Model, const char* Path, int Order, uint64_t MaxLines);

// 이미 만들어둔 토큰 배열로 모델을 만든다. (A9 에서 추가)
//
// Tokens 안에서 TOKEN_EOS 가 문장의 끝을 뜻한다. BOS 패딩은 여기서 붙인다.
// 파일에서 읽는 NgramBuild 와 세는 방법이 똑같다 — 토큰을 주는 쪽만 다르다.
int NgramBuildFromTokens(FNgram* Model, const uint32_t* Tokens, uint64_t Count,
                         int Order);

void NgramFree(FNgram* Model);

// 그램(토큰 Order 개)이 나온 횟수. 없으면 0.
uint64_t NgramCount(const FNgram* Model, const uint32_t* Gram);

// 문맥(토큰 Order-1 개)의 구간 번호. 없으면 -1.
int64_t NgramFindContext(const FNgram* Model, const uint32_t* Context);

// 문맥이 나온 총 횟수. 없으면 0.
uint64_t NgramContextTotal(const FNgram* Model, const uint32_t* Context);

// 문맥 다음에 올 수 있는 토큰의 가짓수.
uint64_t NgramChoiceCount(const FNgram* Model, const uint32_t* Context);

// 문맥 다음에 올 토큰을 확률에 비례해 하나 뽑는다.
// 본 적 없는 문맥이면 TOKEN_EOS.
uint32_t NgramPick(const FNgram* Model, const uint32_t* Context, FRandom* Rng);

// 문장 하나를 만들어 Out 에 담는다. 담은 글자 수를 돌려준다.
int NgramGenerate(const FNgram* Model, FRandom* Rng,
                  uint32_t* Out, int MaxLength);

// ---- 굽기와 불러오기 (A7 에서 추가) ----
//
// 전체 코퍼스로 3그램을 세는 데 30초가 걸린다. 한 번 세고 파일로 구워두면
// 다음부터는 읽기만 하면 된다.

#define NGRAM_MAGIC   "HGNG"        // 파일 앞 4바이트
#define NGRAM_VERSION 1u
#define NGRAM_ORDERMARK 0x01020304u // 바이트 순서 확인용

// 모델을 파일로 쓴다. 성공하면 1.
int NgramSave(const FNgram* Model, const char* Path);

// 파일에서 모델을 읽는다. 성공하면 1.
// 표시가 안 맞거나(다른 기계에서 구웠거나) 버전이 다르면 0.
int NgramLoad(FNgram* Model, const char* Path);

#ifdef __cplusplus
}
#endif
#endif

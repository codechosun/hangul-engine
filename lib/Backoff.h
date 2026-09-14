// lib/Backoff.h
//
// 긴 문맥에서 못 찾으면 짧은 문맥으로 물러서는 모델.
//
// A6 의 N-그램은 본 적 없는 문맥을 만나면 포기한다.
//
//     if (Index < 0) return TOKEN_EOS;   // 문장을 닫는다
//
// N 을 올릴수록 이런 일이 자주 생긴다. 백오프는 대신 이렇게 한다.
//
//     6글자 문맥에서 못 찾으면 -> 5글자로 다시 묻는다
//     5글자에서도 못 찾으면    -> 4글자로 다시 묻는다
//     ...
//     끝까지 가면 글자 빈도(1그램)가 받아준다
//
// "짧은 문맥으로 같은 질문을 다시 한다"는 말 자체가 재귀다.
//
// 못 찾을 때만 물러서는 것이 아니다. 찾았더라도 **확률의 일부를 떼어**
// 짧은 문맥에 넘긴다. 한 번밖에 안 본 문맥을 100% 로 믿지 않기 위해서다.
// 이것을 할인(discounting)이라고 부른다.

#ifndef BACKOFF_H
#define BACKOFF_H

#include <stdint.h>

#include "Ngram.h"
#include "Random.h"

#ifdef __cplusplus
extern "C" {
#endif

// 할인율을 분수로 둔다. 3/4 = 0.75 는 실무에서 흔히 쓰는 값이다.
//
// 소수를 쓰지 않는 이유는 하나다. 정수만으로 하면 **어느 기계에서나
// 같은 수열이 나온다.** 부동소수점은 반올림이 끼어들 수 있다.
#define BACKOFF_NUM 3u
#define BACKOFF_DEN 4u

typedef struct
{
    int MaxOrder;
    FNgram Orders[NGRAM_MAX_ORDER + 1];   // Orders[k] 는 k그램 모델

    // 할인율. 기본값은 BACKOFF_NUM / BACKOFF_DEN 이고,
    // B2 에서 이 값을 바꿔가며 퍼플렉서티를 잰다.
    uint32_t DiscountNum;
    uint32_t DiscountDen;

    // 통계. 어느 차수에서 뽑았는지 센다.
    uint64_t PickCount;
    uint64_t UsedAt[NGRAM_MAX_ORDER + 1];
    uint64_t MissSteps;      // 문맥 자체가 없어서 물러선 횟수
    uint64_t DiscountSteps;  // 할인된 몫에 걸려서 물러선 횟수
} FBackoff;

// 1그램부터 MaxOrder 그램까지 전부 만든다. 성공하면 1.
int BackoffBuild(FBackoff* Backoff, const char* Path,
                 int MaxOrder, uint64_t MaxLines);

void BackoffFree(FBackoff* Backoff);

void BackoffResetStats(FBackoff* Backoff);

// 할인율을 바꾼다. Den 은 0 이면 안 되고 Num < Den 이어야 한다.
void BackoffSetDiscount(FBackoff* Backoff, uint32_t Num, uint32_t Den);

// 문맥(토큰 Length 개) 다음에 올 토큰 하나를 뽑는다.
// 절대 실패하지 않는다. 최악의 경우 글자 빈도에서 뽑는다.
uint32_t BackoffPick(FBackoff* Backoff, const uint32_t* Context, int Length,
                     FRandom* Rng);

// 문장 하나를 만든다. 담은 글자 수를 돌려준다.
int BackoffGenerate(FBackoff* Backoff, FRandom* Rng,
                    uint32_t* Out, int MaxLength);

// 문맥 다음에 Token 이 올 확률. (B2 에서 추가)
//
// BackoffPick 이 뽑는 것과 **정확히 같은 분포**다. 뽑기는 난수로 한 칸을
// 고르는 일이고, 이것은 그 칸의 너비를 재는 일이다.
//
// 1그램도 모르는 토큰이면 0 을 돌려준다. 로그를 씌우면 음의 무한대가 되므로,
// 부르는 쪽에서 반드시 처리해야 한다. B2 본문 참고.
double BackoffProb(const FBackoff* Backoff, const uint32_t* Context, int Length,
                   uint32_t Token);

#ifdef __cplusplus
}
#endif
#endif

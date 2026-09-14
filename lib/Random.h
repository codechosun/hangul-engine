// lib/Random.h
//
// 난수 생성기. xorshift64 를 직접 구현한다.
//
// 표준 rand() 를 쓰지 않는 이유
//   1. MSVC 의 RAND_MAX 는 32767 이다. 큰 범위에서 뽑을 수가 없다.
//   2. 구현마다 결과가 달라서 교재의 수치를 독자가 재현할 수 없다.
//
// 직접 만들면 어느 환경에서든 같은 씨앗에서 같은 수열이 나온다.

#ifndef RANDOM_H
#define RANDOM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    uint64_t State;   // 0 이 되면 안 된다. 계속 0 만 나온다.
} FRandom;

// 씨앗을 심는다. 같은 씨앗은 항상 같은 수열을 만든다.
void RandomSeed(FRandom* Rng, uint64_t Seed);

// 64비트 난수 하나.
uint64_t RandomNext(FRandom* Rng);

// 0 이상 Limit 미만의 난수. Limit 이 0 이면 0.
uint64_t RandomBelow(FRandom* Rng, uint64_t Limit);

#ifdef __cplusplus
}
#endif
#endif

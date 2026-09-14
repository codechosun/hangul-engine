// lib/Sample.h
//
// 빈도표에서 확률에 비례하여 글자를 뽑는다.

#ifndef SAMPLE_H
#define SAMPLE_H

#include <stdint.h>

#include "Freq.h"
#include "Random.h"

#ifdef __cplusplus
extern "C" {
#endif

// 누적 분포를 미리 만들어두고 이진 탐색으로 뽑는다.
//
// 매번 처음부터 더해가며 찾으면 한 번 뽑는 데 O(N) 이 든다.
// 누적합을 한 번 만들어두면 O(log N) 으로 줄어든다.
typedef struct
{
    const FCharCount* Table;   // 빌려 쓴다. 소유하지 않는다.
    uint64_t* Cumulative;      // Cumulative[i] = Table[0..i] 빈도의 합
    int Count;
    uint64_t Total;
} FSampler;

// 빈도표로부터 누적 분포를 만든다. Table 은 살아 있어야 한다.
// 성공하면 1, 메모리가 모자라면 0.
int SamplerInit(FSampler* Sampler, const FCharCount* Table, int Count);

void SamplerFree(FSampler* Sampler);

// 0 이상 Total 미만의 값 하나를 주면, 그 값이 떨어지는 칸의 코드포인트를 준다.
// 난수를 쓰지 않으므로 "어떤 값이 어떤 글자로 가는지" 확인할 때 쓸 수 있다.
uint32_t SamplerFind(const FSampler* Sampler, uint64_t Target);

// 확률에 비례해 코드포인트 하나를 뽑는다.
uint32_t SamplerPick(const FSampler* Sampler, FRandom* Rng);

#ifdef __cplusplus
}
#endif
#endif

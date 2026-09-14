// lib/Sample.c

#include "Sample.h"

#include <assert.h>
#include <stdlib.h>

int SamplerInit(FSampler* Sampler, const FCharCount* Table, int Count)
{
    assert(Sampler != NULL);
    assert(Table != NULL);
    assert(Count > 0);

    uint64_t* Cumulative = (uint64_t*)malloc((size_t)Count * sizeof(uint64_t));
    if (Cumulative == NULL)
    {
        return 0;
    }

    uint64_t Running = 0;
    for (int i = 0; i < Count; i++)
    {
        Running += Table[i].Count;
        Cumulative[i] = Running;
    }

    Sampler->Table = Table;
    Sampler->Cumulative = Cumulative;
    Sampler->Count = Count;
    Sampler->Total = Running;

    return 1;
}

void SamplerFree(FSampler* Sampler)
{
    if (Sampler == NULL)
    {
        return;
    }

    free(Sampler->Cumulative);
    Sampler->Cumulative = NULL;
    Sampler->Table = NULL;
    Sampler->Count = 0;
    Sampler->Total = 0;
}

uint32_t SamplerFind(const FSampler* Sampler, uint64_t Target)
{
    assert(Sampler != NULL);
    assert(Sampler->Cumulative != NULL);

    // Cumulative[i] > Target 을 만족하는 가장 왼쪽 i 를 찾는다.
    //
    // 교재의 이진 탐색과 목표가 다르다. 거기서는 "정확히 같은 값"을 찾았지만
    // 여기서는 그런 값이 아예 없을 수도 있다. 우리가 원하는 것은
    // "Target 을 처음으로 넘어서는 자리" 다.
    int Low = 0;
    int High = Sampler->Count - 1;

    while (Low < High)
    {
        int Mid = Low + (High - Low) / 2;   // (Low+High)/2 는 넘칠 수 있다

        if (Sampler->Cumulative[Mid] > Target)
        {
            High = Mid;
        }
        else
        {
            Low = Mid + 1;
        }
    }

    return Sampler->Table[Low].Codepoint;
}

uint32_t SamplerPick(const FSampler* Sampler, FRandom* Rng)
{
    // 0 이상 Total 미만에서 하나 뽑아, 그 값이 떨어지는 칸을 돌려준다.
    return SamplerFind(Sampler, RandomBelow(Rng, Sampler->Total));
}

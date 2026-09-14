// lib/GramMap.c

#include "GramMap.h"
#include "Map.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define GRAM_MAX_LOAD 0.7

uint64_t GramHash(const uint32_t* Key, int Length)
{
    // 앞에서부터 굴리며 섞는다.
    //
    // 자료구조 교재의 65599 해시가 문자열에 대해 하던 일과 같다.
    //     HashValue = 65599 * HashValue + InKey[i];
    // 곱하는 상수만 64비트에 맞게 키우고, 마지막에 한 번 더 섞는다.
    uint64_t Hash = 0xCBF29CE484222325ull;

    for (int i = 0; i < Length; i++)
    {
        Hash ^= (uint64_t)Key[i];
        Hash *= 0x100000001B3ull;
    }

    return MapHash(Hash);
}

static uint64_t RoundUpPowerOfTwo(uint64_t N)
{
    uint64_t Result = 16;
    while (Result < N)
    {
        Result <<= 1;
    }
    return Result;
}

int GramMapInit(FGramMap* Map, int KeyLength, uint64_t InitialCapacity)
{
    assert(Map != NULL);
    assert(KeyLength > 0);

    memset(Map, 0, sizeof(*Map));

    uint64_t Capacity = RoundUpPowerOfTwo(InitialCapacity);

    Map->KeyLength = KeyLength;
    Map->KeyCapacity = Capacity;
    Map->Keys = (uint32_t*)malloc((size_t)Capacity * (size_t)KeyLength
                                  * sizeof(uint32_t));
    Map->Index = (uint64_t*)calloc((size_t)Capacity, sizeof(uint64_t));
    Map->Values = (uint64_t*)malloc((size_t)Capacity * sizeof(uint64_t));

    if (Map->Keys == NULL || Map->Index == NULL || Map->Values == NULL)
    {
        GramMapFree(Map);
        return 0;
    }

    Map->Capacity = Capacity;
    return 1;
}

void GramMapFree(FGramMap* Map)
{
    if (Map == NULL)
    {
        return;
    }

    free(Map->Keys);
    free(Map->Index);
    free(Map->Values);

    int KeyLength = Map->KeyLength;
    memset(Map, 0, sizeof(*Map));
    Map->KeyLength = KeyLength;
}

const uint32_t* GramMapKeyAt(const FGramMap* Map, uint64_t Index)
{
    assert(Map != NULL);
    assert(Index < Map->KeyCount);

    return Map->Keys + Index * (uint64_t)Map->KeyLength;
}

static int SameKey(const FGramMap* Map, uint64_t Index, const uint32_t* Key)
{
    const uint32_t* Stored = Map->Keys + Index * (uint64_t)Map->KeyLength;
    return memcmp(Stored, Key, (size_t)Map->KeyLength * sizeof(uint32_t)) == 0;
}

static int GrowTable(FGramMap* Map)
{
    uint64_t NewCapacity = Map->Capacity * 2;

    uint64_t* Index = (uint64_t*)calloc((size_t)NewCapacity, sizeof(uint64_t));
    uint64_t* Values = (uint64_t*)malloc((size_t)NewCapacity * sizeof(uint64_t));

    if (Index == NULL || Values == NULL)
    {
        free(Index);
        free(Values);
        return 0;
    }

    uint64_t Mask = NewCapacity - 1;

    // 아레나는 건드리지 않는다. 자리표만 다시 만든다.
    for (uint64_t i = 0; i < Map->Capacity; i++)
    {
        if (Map->Index[i] == 0)
        {
            continue;
        }

        uint64_t KeyIndex = Map->Index[i] - 1;
        const uint32_t* Key = Map->Keys + KeyIndex * (uint64_t)Map->KeyLength;

        uint64_t Slot = GramHash(Key, Map->KeyLength) & Mask;
        while (Index[Slot] != 0)
        {
            Slot = (Slot + 1) & Mask;
        }

        Index[Slot] = Map->Index[i];
        Values[Slot] = Map->Values[i];
    }

    free(Map->Index);
    free(Map->Values);

    Map->Index = Index;
    Map->Values = Values;
    Map->Capacity = NewCapacity;

    return 1;
}

static int GrowArena(FGramMap* Map)
{
    uint64_t NewCapacity = Map->KeyCapacity * 2;

    uint32_t* Keys = (uint32_t*)realloc(
        Map->Keys, (size_t)NewCapacity * (size_t)Map->KeyLength * sizeof(uint32_t));

    if (Keys == NULL)
    {
        return 0;
    }

    // realloc 이 성공하면 옛 주소는 무효다.
    // 우리는 포인터가 아니라 번호를 저장했으므로 고칠 것이 없다.
    Map->Keys = Keys;
    Map->KeyCapacity = NewCapacity;

    return 1;
}

int GramMapAdd(FGramMap* Map, const uint32_t* Key, uint64_t Delta)
{
    assert(Map != NULL);
    assert(Key != NULL);

    if ((double)(Map->Count + 1) > (double)Map->Capacity * GRAM_MAX_LOAD)
    {
        if (!GrowTable(Map))
        {
            return 0;
        }
    }

    uint64_t Mask = Map->Capacity - 1;
    uint64_t Slot = GramHash(Key, Map->KeyLength) & Mask;

    Map->CallCount++;

    for (;;)
    {
        Map->ProbeCount++;

        if (Map->Index[Slot] == 0)
        {
            if (Map->KeyCount == Map->KeyCapacity && !GrowArena(Map))
            {
                return 0;
            }

            uint64_t KeyIndex = Map->KeyCount++;
            memcpy(Map->Keys + KeyIndex * (uint64_t)Map->KeyLength,
                   Key, (size_t)Map->KeyLength * sizeof(uint32_t));

            Map->Index[Slot] = KeyIndex + 1;   // 0 은 "빈 칸" 이므로 1 을 더한다
            Map->Values[Slot] = Delta;
            Map->Count++;
            return 1;
        }

        if (SameKey(Map, Map->Index[Slot] - 1, Key))
        {
            Map->Values[Slot] += Delta;
            return 1;
        }

        Slot = (Slot + 1) & Mask;
    }
}

uint64_t GramMapGet(const FGramMap* Map, const uint32_t* Key)
{
    assert(Map != NULL);
    assert(Key != NULL);

    uint64_t Mask = Map->Capacity - 1;
    uint64_t Slot = GramHash(Key, Map->KeyLength) & Mask;

    for (;;)
    {
        if (Map->Index[Slot] == 0)
        {
            return 0;
        }

        if (SameKey(Map, Map->Index[Slot] - 1, Key))
        {
            return Map->Values[Slot];
        }

        Slot = (Slot + 1) & Mask;
    }
}

double GramMapAverageProbe(const FGramMap* Map)
{
    if (Map == NULL || Map->CallCount == 0)
    {
        return 0.0;
    }

    return (double)Map->ProbeCount / (double)Map->CallCount;
}

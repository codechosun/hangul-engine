// lib/Map.c

#include "Map.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

// 이 비율을 넘으면 칸을 두 배로 늘린다.
// 0.7 을 고른 이유는 A4 본문에 있다.
#define MAP_MAX_LOAD 0.7

uint64_t MapHash(uint64_t Key)
{
    // 곱하고 섞기를 반복한다. 65599 해시와 발상이 같다.
    // 다른 점은 상수가 64비트에 맞게 크고, 섞는 횟수가 많다는 것뿐이다.
    uint64_t X = Key;

    X ^= X >> 33;
    X *= 0xFF51AFD7ED558CCDull;
    X ^= X >> 33;
    X *= 0xC4CEB9FE1A85EC53ull;
    X ^= X >> 33;

    return X;
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

int MapInit(FMap* Map, uint64_t InitialCapacity)
{
    assert(Map != NULL);

    uint64_t Capacity = RoundUpPowerOfTwo(InitialCapacity);

    Map->Keys = (uint64_t*)malloc((size_t)Capacity * sizeof(uint64_t));
    Map->Values = (uint64_t*)malloc((size_t)Capacity * sizeof(uint64_t));
    Map->Used = (uint8_t*)calloc((size_t)Capacity, sizeof(uint8_t));

    if (Map->Keys == NULL || Map->Values == NULL || Map->Used == NULL)
    {
        MapFree(Map);
        return 0;
    }

    Map->Capacity = Capacity;
    Map->Count = 0;
    Map->ProbeCount = 0;
    Map->CallCount = 0;

    return 1;
}

void MapFree(FMap* Map)
{
    if (Map == NULL)
    {
        return;
    }

    free(Map->Keys);
    free(Map->Values);
    free(Map->Used);

    Map->Keys = NULL;
    Map->Values = NULL;
    Map->Used = NULL;
    Map->Capacity = 0;
    Map->Count = 0;
}

// 통계를 세지 않는 삽입. 리해싱 중에 쓴다.
// 새 표에는 중복이 있을 수 없으므로 "이미 있는가" 검사도 필요 없다.
static void InsertFresh(FMap* Map, uint64_t Key, uint64_t Value)
{
    uint64_t Mask = Map->Capacity - 1;
    uint64_t Slot = MapHash(Key) & Mask;

    while (Map->Used[Slot] != 0)
    {
        Slot = (Slot + 1) & Mask;
    }

    Map->Keys[Slot] = Key;
    Map->Values[Slot] = Value;
    Map->Used[Slot] = 1;
}

static int Grow(FMap* Map)
{
    FMap Bigger;
    if (!MapInit(&Bigger, Map->Capacity * 2))
    {
        return 0;
    }

    for (uint64_t i = 0; i < Map->Capacity; i++)
    {
        if (Map->Used[i] != 0)
        {
            InsertFresh(&Bigger, Map->Keys[i], Map->Values[i]);
        }
    }

    Bigger.Count = Map->Count;
    Bigger.ProbeCount = Map->ProbeCount;
    Bigger.CallCount = Map->CallCount;

    free(Map->Keys);
    free(Map->Values);
    free(Map->Used);
    *Map = Bigger;

    return 1;
}

int MapAdd(FMap* Map, uint64_t Key, uint64_t Delta)
{
    assert(Map != NULL);
    assert(Map->Used != NULL);

    if ((double)(Map->Count + 1) > (double)Map->Capacity * MAP_MAX_LOAD)
    {
        if (!Grow(Map))
        {
            return 0;
        }
    }

    uint64_t Mask = Map->Capacity - 1;
    uint64_t Slot = MapHash(Key) & Mask;

    Map->CallCount++;

    for (;;)
    {
        Map->ProbeCount++;

        if (Map->Used[Slot] == 0)
        {
            Map->Keys[Slot] = Key;
            Map->Values[Slot] = Delta;
            Map->Used[Slot] = 1;
            Map->Count++;
            return 1;
        }

        if (Map->Keys[Slot] == Key)
        {
            Map->Values[Slot] += Delta;
            return 1;
        }

        Slot = (Slot + 1) & Mask;
    }
}

uint64_t MapGet(const FMap* Map, uint64_t Key)
{
    assert(Map != NULL);
    assert(Map->Used != NULL);

    uint64_t Mask = Map->Capacity - 1;
    uint64_t Slot = MapHash(Key) & Mask;

    for (;;)
    {
        if (Map->Used[Slot] == 0)
        {
            return 0;
        }

        if (Map->Keys[Slot] == Key)
        {
            return Map->Values[Slot];
        }

        Slot = (Slot + 1) & Mask;
    }
}

double MapAverageProbe(const FMap* Map)
{
    if (Map == NULL || Map->CallCount == 0)
    {
        return 0.0;
    }

    return (double)Map->ProbeCount / (double)Map->CallCount;
}

double MapLoadFactor(const FMap* Map)
{
    if (Map == NULL || Map->Capacity == 0)
    {
        return 0.0;
    }

    return (double)Map->Count / (double)Map->Capacity;
}

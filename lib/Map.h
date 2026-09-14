// lib/Map.h
//
// 64비트 정수 키 -> 64비트 카운터 해시맵.
//
// 자료구조 교재 1.7 의 해시맵을 이 교재의 용도에 맞게 뜯어고친 것이다.
// 바뀐 곳은 네 군데.
//
//   1. 키가 문자열이 아니라 uint64_t 다      -> strcmp 가 사라진다
//   2. 칸 수가 고정이 아니라 자란다          -> 리해싱이 생긴다
//   3. 칸 수가 소수가 아니라 2의 거듭제곱    -> % 가 & 로 바뀐다
//   4. 값을 덮어쓰지 않고 더한다             -> 세는 것이 목적이므로
//
// 지우는 기능은 없다. 세기만 하기 때문에 필요가 없고,
// 없으면 DELETED 상태를 관리할 필요도 없어진다.

#ifndef MAP_H
#define MAP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    uint64_t* Keys;
    uint64_t* Values;
    uint8_t*  Used;        // 0 이면 빈 칸

    uint64_t  Capacity;    // 항상 2의 거듭제곱
    uint64_t  Count;       // 실제로 들어 있는 개수

    uint64_t  ProbeCount;  // 통계용. 지금까지 칸을 몇 번 들여다봤는가
    uint64_t  CallCount;   // 통계용. MapAdd/MapGet 을 몇 번 불렀는가
} FMap;

// 키를 골고루 흩뜨린다. 자료구조 교재의 65599 해시를 64비트로 키운 것.
uint64_t MapHash(uint64_t Key);

// InitialCapacity 는 2의 거듭제곱으로 올림된다. 최소 16.
// 성공하면 1, 메모리가 모자라면 0.
int MapInit(FMap* Map, uint64_t InitialCapacity);

void MapFree(FMap* Map);

// Key 의 값에 Delta 를 더한다. 없던 키면 새로 만든다.
// 성공하면 1, 자리를 늘리다 메모리가 모자라면 0.
int MapAdd(FMap* Map, uint64_t Key, uint64_t Delta);

// Key 의 값. 없으면 0.
uint64_t MapGet(const FMap* Map, uint64_t Key);

// 한 번 찾는 데 평균 몇 칸을 들여다봤는가.
double MapAverageProbe(const FMap* Map);

// 얼마나 찼는가. 0.0 ~ 1.0
double MapLoadFactor(const FMap* Map);

#ifdef __cplusplus
}
#endif
#endif

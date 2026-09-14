// lib/GramMap.h
//
// 토큰 여러 개를 키로 쓰는 해시맵.
//
// A4 의 lib/Map 은 키가 uint64_t 하나였다. 글자 두 개까지는 그걸로 됐다.
// 세 개부터는 안 된다. 코드포인트 하나가 21비트이므로
//
//     3개 = 63비트  (아슬아슬하게 들어간다)
//     4개 = 84비트  (넘친다)
//
// 그래서 키를 **값**이 아니라 **배열**로 다룬다. 바뀌는 것이 셋이다.
//
//   1. 키를 어딘가에 담아둬야 한다        -> 아레나(arena)
//   2. 비교가 == 이 아니라 한 칸씩 비교다 -> memcmp
//   3. 해시도 여러 값을 섞어야 한다       -> 굴리며 섞기
//
// 키 길이는 하나의 맵 안에서 항상 같다. 그래서 길이를 따로 저장하지 않는다.

#ifndef GRAMMAP_H
#define GRAMMAP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    int KeyLength;          // 키 하나에 들어가는 토큰 수

    // --- 아레나. 키의 실제 내용이 여기 산다 ---
    uint32_t* Keys;         // KeyCount * KeyLength 개의 토큰
    uint64_t  KeyCount;
    uint64_t  KeyCapacity;  // 담을 수 있는 키의 개수

    // --- 열린 주소 해시표. 아레나를 가리키기만 한다 ---
    uint64_t* Index;        // 0 이면 빈 칸, 아니면 (키 번호 + 1)
    uint64_t* Values;
    uint64_t  Capacity;     // 항상 2의 거듭제곱
    uint64_t  Count;

    uint64_t  ProbeCount;
    uint64_t  CallCount;
} FGramMap;

// 토큰 여러 개를 64비트 하나로 섞는다.
uint64_t GramHash(const uint32_t* Key, int Length);

int GramMapInit(FGramMap* Map, int KeyLength, uint64_t InitialCapacity);

void GramMapFree(FGramMap* Map);

// Key 의 값에 Delta 를 더한다. 없던 키면 새로 만든다. 성공하면 1.
int GramMapAdd(FGramMap* Map, const uint32_t* Key, uint64_t Delta);

// Key 의 값. 없으면 0.
uint64_t GramMapGet(const FGramMap* Map, const uint32_t* Key);

// Index 번째 키가 시작하는 곳.
//
// **주의.** 이 포인터는 다음 GramMapAdd 까지만 유효하다.
// 아레나가 자라면서 realloc 되면 주소가 통째로 바뀐다.
// 오래 들고 있어야 하면 포인터가 아니라 번호를 저장할 것.
const uint32_t* GramMapKeyAt(const FGramMap* Map, uint64_t Index);

double GramMapAverageProbe(const FGramMap* Map);

#ifdef __cplusplus
}
#endif
#endif

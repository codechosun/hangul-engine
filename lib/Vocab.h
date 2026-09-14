// lib/Vocab.h
//
// 문자열을 번호로 바꾸는 사전.
//
// A1~A8 은 토큰이 곧 코드포인트여서 번호가 이미 있었다.
// 형태소를 다루기 시작하면 토큰이 "한국어" 같은 **문자열**이 된다.
// N-그램 코드는 uint32_t 토큰만 아므로, 문자열마다 번호를 붙여줘야 한다.
//
// 이 교재에서 만드는 **세 번째 해시맵**이다.
//
//   lib/Map      키가 uint64_t 하나       (A4)
//   lib/GramMap  키가 uint32_t 배열       (A6)
//   lib/Vocab    키가 문자열              (A9)
//
// 셋 다 열린 주소·선형 탐사·2의 거듭제곱 칸으로 똑같이 생겼다.
// 다른 것은 "키를 어떻게 비교하고 어떻게 섞는가" 뿐이다.
// C 에서는 이 셋을 하나로 합칠 방법이 마땅치 않다. B1 에서 그 이야기를 한다.

#ifndef VOCAB_H
#define VOCAB_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    // --- 아레나. 문자열 내용이 여기 산다 ---
    char*     Text;         // 모든 문자열을 '\0' 로 끊어 이어 붙인 것
    uint64_t  TextUsed;
    uint64_t  TextCapacity;

    // --- 번호 -> 아레나 안의 위치 ---
    uint64_t* Offsets;
    uint64_t  Count;        // 등록된 문자열의 개수 = 다음에 줄 번호
    uint64_t  OffsetCapacity;

    // --- 열린 주소 해시표. 번호만 담는다 ---
    uint32_t* Slots;        // 0 이면 빈 칸, 아니면 (번호 + 1)
    uint64_t  SlotCapacity;
} FVocab;

// 문자열 하나를 64비트로 섞는다. 자료구조 교재 1.7 의 65599 해시를 키운 것.
uint64_t VocabHash(const char* Text);

int VocabInit(FVocab* Vocab, uint64_t InitialCapacity);

void VocabFree(FVocab* Vocab);

// 없으면 새로 등록하고, 있으면 이미 있던 번호를 준다.
// 실패하면 0xFFFFFFFF.
uint32_t VocabIntern(FVocab* Vocab, const char* Text);

// 등록된 번호를 찾는다. 없으면 0xFFFFFFFF.
uint32_t VocabFind(const FVocab* Vocab, const char* Text);

// 번호에 해당하는 문자열. 범위 밖이면 NULL.
//
// **주의.** 다음 VocabIntern 까지만 유효하다. 아레나가 자라면 주소가 바뀐다.
const char* VocabText(const FVocab* Vocab, uint32_t Id);

#define VOCAB_NONE 0xFFFFFFFFu

#ifdef __cplusplus
}
#endif
#endif

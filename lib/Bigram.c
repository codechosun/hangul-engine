// lib/Bigram.c

#include "Bigram.h"
#include "Map.h"
#include "Scan.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

// 표를 만드는 동안만 쓰는 임시 형태. 정렬해서 CSR 로 옮긴다.
typedef struct
{
    uint32_t Prev;
    uint32_t Next;
    uint64_t Count;
} FPairRow;

static uint64_t PackPair(uint32_t Prev, uint32_t Next)
{
    return ((uint64_t)Prev << 32) | (uint64_t)Next;
}

// 앞 글자 오름차순, 같으면 뒤 글자 오름차순.
//
// 2순위가 없으면 해시맵을 훑은 순서가 그대로 남아서,
// 표의 칸 수가 조금만 달라져도 결과 파일이 달라진다.
// 파이썬 참조 구현과 한 줄씩 맞추려면 순서가 정해져 있어야 한다.
static int ComparePairRow(const void* A, const void* B)
{
    const FPairRow* Left = (const FPairRow*)A;
    const FPairRow* Right = (const FPairRow*)B;

    if (Left->Prev < Right->Prev) return -1;
    if (Left->Prev > Right->Prev) return 1;
    if (Left->Next < Right->Next) return -1;
    if (Left->Next > Right->Next) return 1;
    return 0;
}

int BigramBuild(FBigram* Model, const char* Path)
{
    assert(Model != NULL);
    assert(Path != NULL);

    memset(Model, 0, sizeof(*Model));

    FMap Counts;
    if (!MapInit(&Counts, 1024))
    {
        return 0;
    }

    FScanner Scanner;
    if (!ScanOpen(&Scanner, Path))
    {
        MapFree(&Counts);
        return 0;
    }

    // ---- 1단계. 쌍을 센다 ----
    //
    // 줄바꿈은 글자로 세지 않는다. "여기서 문장이 끝나고 다음이 시작된다"는
    // 표시일 뿐이므로 EOS 와 BOS 로 갈아 끼운다.
    uint64_t Total = 0;
    uint64_t Lines = 0;
    uint32_t Prev = TOKEN_BOS;
    uint32_t Code = 0;

    while (ScanNext(&Scanner, &Code))
    {
        if (Code == (uint32_t)'\n')
        {
            MapAdd(&Counts, PackPair(Prev, TOKEN_EOS), 1);
            Total++;
            Lines++;
            Prev = TOKEN_BOS;
        }
        else
        {
            MapAdd(&Counts, PackPair(Prev, Code), 1);
            Total++;
            Prev = Code;
        }
    }

    ScanClose(&Scanner);

    // 파일이 줄바꿈으로 끝나지 않았다면 마지막 문장을 닫아준다.
    if (Prev != TOKEN_BOS)
    {
        MapAdd(&Counts, PackPair(Prev, TOKEN_EOS), 1);
        Total++;
        Lines++;
    }

    // ---- 2단계. 해시맵을 배열로 옮겨 정렬한다 ----
    FPairRow* Rows = (FPairRow*)malloc((size_t)Counts.Count * sizeof(FPairRow));
    if (Rows == NULL)
    {
        MapFree(&Counts);
        return 0;
    }

    uint64_t RowCount = 0;
    for (uint64_t i = 0; i < Counts.Capacity; i++)
    {
        if (Counts.Used[i] != 0)
        {
            Rows[RowCount].Prev = (uint32_t)(Counts.Keys[i] >> 32);
            Rows[RowCount].Next = (uint32_t)(Counts.Keys[i] & 0xFFFFFFFFu);
            Rows[RowCount].Count = Counts.Values[i];
            RowCount++;
        }
    }

    MapFree(&Counts);

    qsort(Rows, (size_t)RowCount, sizeof(FPairRow), ComparePairRow);

    // ---- 3단계. CSR 로 옮긴다 ----
    uint32_t* Start = (uint32_t*)calloc(VOCAB_LIMIT + 1, sizeof(uint32_t));
    uint32_t* Next = (uint32_t*)malloc((size_t)RowCount * sizeof(uint32_t));
    uint64_t* Cumulative = (uint64_t*)malloc((size_t)RowCount * sizeof(uint64_t));

    if (Start == NULL || Next == NULL || Cumulative == NULL)
    {
        free(Start);
        free(Next);
        free(Cumulative);
        free(Rows);
        return 0;
    }

    // 앞 글자별 개수를 세어 누적하면 각 구간의 시작 위치가 나온다.
    // 이 방식이면 한 번도 안 나온 앞 글자도 자동으로 "시작 == 끝" 이 된다.
    for (uint64_t i = 0; i < RowCount; i++)
    {
        Start[Rows[i].Prev + 1]++;
    }

    for (uint32_t i = 0; i < VOCAB_LIMIT; i++)
    {
        Start[i + 1] += Start[i];
    }

    // 행이 (앞, 뒤) 순으로 정렬되어 있으므로 순서대로 담기만 하면 된다.
    uint64_t Running = 0;
    uint32_t Current = 0xFFFFFFFFu;

    for (uint64_t i = 0; i < RowCount; i++)
    {
        if (Rows[i].Prev != Current)
        {
            Current = Rows[i].Prev;
            Running = 0;
        }

        Running += Rows[i].Count;
        Next[i] = Rows[i].Next;
        Cumulative[i] = Running;
    }

    free(Rows);

    Model->Next = Next;
    Model->Cumulative = Cumulative;
    Model->Start = Start;
    Model->PairCount = RowCount;
    Model->Total = Total;
    Model->LineCount = Lines;

    return 1;
}

void BigramFree(FBigram* Model)
{
    if (Model == NULL)
    {
        return;
    }

    free(Model->Next);
    free(Model->Cumulative);
    free(Model->Start);

    memset(Model, 0, sizeof(*Model));
}

uint64_t BigramCount(const FBigram* Model, uint32_t Prev, uint32_t Next)
{
    assert(Model != NULL);

    if (Prev >= VOCAB_LIMIT)
    {
        return 0;
    }

    uint32_t From = Model->Start[Prev];
    uint32_t To = Model->Start[Prev + 1];

    // 구간 안은 뒤 글자 오름차순이므로 이진 탐색이 된다.
    while (From < To)
    {
        uint32_t Mid = From + (To - From) / 2;

        if (Model->Next[Mid] < Next)
        {
            From = Mid + 1;
        }
        else
        {
            To = Mid;
        }
    }

    if (From >= Model->Start[Prev + 1] || Model->Next[From] != Next)
    {
        return 0;
    }

    uint64_t Before = (From == Model->Start[Prev]) ? 0 : Model->Cumulative[From - 1];
    return Model->Cumulative[From] - Before;
}

uint64_t BigramContextTotal(const FBigram* Model, uint32_t Prev)
{
    assert(Model != NULL);

    if (Prev >= VOCAB_LIMIT)
    {
        return 0;
    }

    uint32_t From = Model->Start[Prev];
    uint32_t To = Model->Start[Prev + 1];

    if (From == To)
    {
        return 0;
    }

    // 구간의 마지막 누적값이 곧 그 앞 글자가 나온 총 횟수다.
    return Model->Cumulative[To - 1];
}

uint32_t BigramChoiceCount(const FBigram* Model, uint32_t Prev)
{
    assert(Model != NULL);

    if (Prev >= VOCAB_LIMIT)
    {
        return 0;
    }

    return Model->Start[Prev + 1] - Model->Start[Prev];
}

uint32_t BigramPick(const FBigram* Model, uint32_t Prev, FRandom* Rng)
{
    assert(Model != NULL);

    if (Prev >= VOCAB_LIMIT)
    {
        return TOKEN_EOS;
    }

    uint32_t From = Model->Start[Prev];
    uint32_t To = Model->Start[Prev + 1];

    if (From == To)
    {
        return TOKEN_EOS;   // 본 적 없는 앞 글자. 문장을 닫는다.
    }

    uint64_t Target = RandomBelow(Rng, Model->Cumulative[To - 1]);

    // A3 과 같은 lower bound. 다만 배열 전체가 아니라 구간 안에서만 찾는다.
    uint32_t Low = From;
    uint32_t High = To - 1;

    while (Low < High)
    {
        uint32_t Mid = Low + (High - Low) / 2;

        if (Model->Cumulative[Mid] > Target)
        {
            High = Mid;
        }
        else
        {
            Low = Mid + 1;
        }
    }

    return Model->Next[Low];
}

int BigramGenerate(const FBigram* Model, FRandom* Rng,
                   uint32_t* Out, int MaxLength)
{
    assert(Model != NULL);
    assert(Out != NULL);

    uint32_t Prev = TOKEN_BOS;
    int Length = 0;

    while (Length < MaxLength)
    {
        uint32_t Code = BigramPick(Model, Prev, Rng);

        if (Code == TOKEN_EOS)
        {
            break;
        }

        Out[Length++] = Code;
        Prev = Code;
    }

    return Length;
}

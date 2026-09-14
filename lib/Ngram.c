// lib/Ngram.c

#include "Ngram.h"
#include "GramMap.h"
#include "Scan.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

// 정렬하는 동안만 쓰는 형태.
typedef struct
{
    uint32_t Token[NGRAM_MAX_ORDER];
    uint64_t Count;
} FGramRow;

// qsort 의 비교 함수는 인자를 두 개밖에 못 받는다.
// "토큰을 몇 개까지 비교할 것인가"를 넘길 자리가 없다.
//
// 표준 C 에는 방법이 없어서 파일 범위 변수를 쓴다. 보기 좋지 않지만
// C 로 qsort 를 쓰는 한 피할 수 없다.
// (POSIX 의 qsort_r 과 MSVC 의 qsort_s 가 있지만 서로 인자 순서가 다르다.)
//
// C++ 에서는 람다가 이 문제를 없앤다. B파트에서 다시 본다.
static int GSortOrder = 0;

static int CompareGramRow(const void* A, const void* B)
{
    const FGramRow* Left = (const FGramRow*)A;
    const FGramRow* Right = (const FGramRow*)B;

    for (int i = 0; i < GSortOrder; i++)
    {
        if (Left->Token[i] < Right->Token[i]) return -1;
        if (Left->Token[i] > Right->Token[i]) return 1;
    }

    return 0;
}

static int SameContext(const uint32_t* A, const uint32_t* B, int Length)
{
    return memcmp(A, B, (size_t)Length * sizeof(uint32_t)) == 0;
}

// 사전순 비교. memcmp 를 쓰면 안 된다.
//
// memcmp 는 바이트 단위로 비교하는데, 리틀 엔디언에서는 uint32_t 의
// 낮은 바이트가 앞에 놓인다. 0x100 과 0x2 를 비교하면 memcmp 는
// 0x00,0x01 과 0x02,0x00 을 보고 0x2 가 크다고 답한다. 숫자 순서가 아니다.
//
// 같은지만 볼 때는 memcmp 로 충분하다. 순서를 볼 때는 직접 세야 한다.
static int CompareTokens(const uint32_t* A, const uint32_t* B, int Length)
{
    for (int i = 0; i < Length; i++)
    {
        if (A[i] < B[i]) return -1;
        if (A[i] > B[i]) return 1;
    }
    return 0;
}

int NgramBuild(FNgram* Model, const char* Path, int Order, uint64_t MaxLines)
{
    assert(Model != NULL);
    assert(Order >= 1 && Order <= NGRAM_MAX_ORDER);

    memset(Model, 0, sizeof(*Model));
    Model->Order = Order;

    FGramMap Map;
    if (!GramMapInit(&Map, Order, 1024))
    {
        return 0;
    }

    FScanner Scanner;
    if (!ScanOpen(&Scanner, Path))
    {
        GramMapFree(&Map);
        return 0;
    }

    // ---- 1단계. 창을 굴리며 센다 ----
    //
    // 창은 "마지막 Order 개의 토큰" 이다. 한 칸 밀고 뒤에 새 토큰을 넣으면
    // 창 전체가 곧 하나의 그램이 된다.
    //
    // 문장이 시작할 때는 **창 전체**를 BOS 로 채운다.
    // 앞의 Order-1 개만 채우면, 한 칸 미는 순간 맨 뒤에 남아 있던
    // 지난 문장의 EOS 가 문맥 안으로 딸려 들어온다.
    uint32_t Window[NGRAM_MAX_ORDER];
    for (int i = 0; i < Order; i++)
    {
        Window[i] = TOKEN_BOS;
    }

    uint64_t Total = 0;
    uint64_t Lines = 0;
    uint64_t Chars = 0;
    uint32_t Code = 0;

    while (ScanNext(&Scanner, &Code))
    {
        uint32_t Token = (Code == (uint32_t)'\n') ? TOKEN_EOS : Code;

        // 창을 한 칸 민다. Order 가 8 이하라 그냥 옮기는 게 빠르다.
        for (int i = 0; i < Order - 1; i++)
        {
            Window[i] = Window[i + 1];
        }
        Window[Order - 1] = Token;

        GramMapAdd(&Map, Window, 1);
        Total++;

        if (Token == TOKEN_EOS)
        {
            Lines++;

            for (int i = 0; i < Order; i++)
            {
                Window[i] = TOKEN_BOS;
            }

            if (MaxLines != 0 && Lines >= MaxLines)
            {
                break;
            }
        }
        else
        {
            Chars++;
        }
    }

    ScanClose(&Scanner);

    // 파일이 줄바꿈으로 끝나지 않았다면 마지막 문장을 닫는다.
    // 문장을 정상적으로 닫았으면 창이 전부 BOS 이므로 여기 걸리지 않는다.
    if (Window[Order - 1] != TOKEN_BOS)
    {
        for (int i = 0; i < Order - 1; i++)
        {
            Window[i] = Window[i + 1];
        }
        Window[Order - 1] = TOKEN_EOS;

        GramMapAdd(&Map, Window, 1);
        Total++;
        Lines++;
    }

    // ---- 2단계. 꺼내서 사전순으로 정렬 ----
    FGramRow* Rows = (FGramRow*)malloc((size_t)Map.Count * sizeof(FGramRow));
    if (Rows == NULL)
    {
        GramMapFree(&Map);
        return 0;
    }

    uint64_t RowCount = 0;
    uint64_t Once = 0;

    for (uint64_t i = 0; i < Map.Capacity; i++)
    {
        if (Map.Index[i] == 0)
        {
            continue;
        }

        const uint32_t* Key = GramMapKeyAt(&Map, Map.Index[i] - 1);

        memset(&Rows[RowCount], 0, sizeof(FGramRow));
        memcpy(Rows[RowCount].Token, Key, (size_t)Order * sizeof(uint32_t));
        Rows[RowCount].Count = Map.Values[i];

        if (Map.Values[i] == 1)
        {
            Once++;
        }

        RowCount++;
    }

    GramMapFree(&Map);

    GSortOrder = Order;
    qsort(Rows, (size_t)RowCount, sizeof(FGramRow), CompareGramRow);

    // ---- 3단계. 배열로 펴고 문맥 구간을 만든다 ----
    uint32_t* Grams =
        (uint32_t*)malloc((size_t)RowCount * (size_t)Order * sizeof(uint32_t));
    uint64_t* Cumulative =
        (uint64_t*)malloc((size_t)RowCount * sizeof(uint64_t));

    if (Grams == NULL || Cumulative == NULL)
    {
        free(Grams);
        free(Cumulative);
        free(Rows);
        return 0;
    }

    // 문맥이 몇 개인지 먼저 센다.
    uint64_t ContextCount = 0;
    for (uint64_t i = 0; i < RowCount; i++)
    {
        if (i == 0 || !SameContext(Rows[i].Token, Rows[i - 1].Token, Order - 1))
        {
            ContextCount++;
        }
    }

    uint64_t* ContextStart =
        (uint64_t*)malloc((size_t)(ContextCount + 1) * sizeof(uint64_t));

    if (ContextStart == NULL)
    {
        free(Grams);
        free(Cumulative);
        free(Rows);
        return 0;
    }

    uint64_t ContextIndex = 0;
    uint64_t Running = 0;

    for (uint64_t i = 0; i < RowCount; i++)
    {
        if (i == 0 || !SameContext(Rows[i].Token, Rows[i - 1].Token, Order - 1))
        {
            ContextStart[ContextIndex++] = i;
            Running = 0;
        }

        Running += Rows[i].Count;
        Cumulative[i] = Running;

        memcpy(Grams + i * (uint64_t)Order, Rows[i].Token,
               (size_t)Order * sizeof(uint32_t));
    }

    ContextStart[ContextCount] = RowCount;

    free(Rows);

    Model->Grams = Grams;
    Model->Cumulative = Cumulative;
    Model->GramCount = RowCount;
    Model->ContextStart = ContextStart;
    Model->ContextCount = ContextCount;
    Model->Total = Total;
    Model->OnceCount = Once;
    Model->LineCount = Lines;
    Model->CharCount = Chars;

    return 1;
}

void NgramFree(FNgram* Model)
{
    if (Model == NULL)
    {
        return;
    }

    free(Model->Grams);
    free(Model->Cumulative);
    free(Model->ContextStart);

    int Order = Model->Order;
    memset(Model, 0, sizeof(*Model));
    Model->Order = Order;
}

int64_t NgramFindContext(const FNgram* Model, const uint32_t* Context)
{
    assert(Model != NULL);

    int Length = Model->Order - 1;
    if (Length == 0)
    {
        return (Model->ContextCount > 0) ? 0 : -1;
    }

    // 문맥 구간들도 사전순이므로 이진 탐색이 된다.
    uint64_t Low = 0;
    uint64_t High = Model->ContextCount;

    while (Low < High)
    {
        uint64_t Mid = Low + (High - Low) / 2;
        const uint32_t* Here =
            Model->Grams + Model->ContextStart[Mid] * (uint64_t)Model->Order;

        int Compared = CompareTokens(Here, Context, Length);

        if (Compared < 0)
        {
            Low = Mid + 1;
        }
        else
        {
            High = Mid;
        }
    }

    if (Low >= Model->ContextCount)
    {
        return -1;
    }

    const uint32_t* Here =
        Model->Grams + Model->ContextStart[Low] * (uint64_t)Model->Order;

    return SameContext(Here, Context, Length) ? (int64_t)Low : -1;
}

uint64_t NgramCount(const FNgram* Model, const uint32_t* Gram)
{
    assert(Model != NULL);

    int64_t Context = NgramFindContext(Model, Gram);
    if (Context < 0)
    {
        return 0;
    }

    uint64_t From = Model->ContextStart[Context];
    uint64_t To = Model->ContextStart[Context + 1];
    uint32_t Want = Gram[Model->Order - 1];

    uint64_t Low = From;
    uint64_t High = To;

    while (Low < High)
    {
        uint64_t Mid = Low + (High - Low) / 2;
        uint32_t Here = Model->Grams[Mid * (uint64_t)Model->Order + Model->Order - 1];

        if (Here < Want)
        {
            Low = Mid + 1;
        }
        else
        {
            High = Mid;
        }
    }

    if (Low >= To)
    {
        return 0;
    }

    if (Model->Grams[Low * (uint64_t)Model->Order + Model->Order - 1] != Want)
    {
        return 0;
    }

    uint64_t Before = (Low == From) ? 0 : Model->Cumulative[Low - 1];
    return Model->Cumulative[Low] - Before;
}

uint64_t NgramContextTotal(const FNgram* Model, const uint32_t* Context)
{
    int64_t Index = NgramFindContext(Model, Context);
    if (Index < 0)
    {
        return 0;
    }

    return Model->Cumulative[Model->ContextStart[Index + 1] - 1];
}

uint64_t NgramChoiceCount(const FNgram* Model, const uint32_t* Context)
{
    int64_t Index = NgramFindContext(Model, Context);
    if (Index < 0)
    {
        return 0;
    }

    return Model->ContextStart[Index + 1] - Model->ContextStart[Index];
}

uint32_t NgramPick(const FNgram* Model, const uint32_t* Context, FRandom* Rng)
{
    int64_t Index = NgramFindContext(Model, Context);
    if (Index < 0)
    {
        return TOKEN_EOS;
    }

    uint64_t From = Model->ContextStart[Index];
    uint64_t To = Model->ContextStart[Index + 1];

    uint64_t Target = RandomBelow(Rng, Model->Cumulative[To - 1]);

    uint64_t Low = From;
    uint64_t High = To - 1;

    while (Low < High)
    {
        uint64_t Mid = Low + (High - Low) / 2;

        if (Model->Cumulative[Mid] > Target)
        {
            High = Mid;
        }
        else
        {
            Low = Mid + 1;
        }
    }

    return Model->Grams[Low * (uint64_t)Model->Order + Model->Order - 1];
}

int NgramGenerate(const FNgram* Model, FRandom* Rng,
                  uint32_t* Out, int MaxLength)
{
    assert(Model != NULL);
    assert(Out != NULL);

    uint32_t Context[NGRAM_MAX_ORDER];
    for (int i = 0; i < Model->Order - 1; i++)
    {
        Context[i] = TOKEN_BOS;
    }

    int Length = 0;

    while (Length < MaxLength)
    {
        uint32_t Token = NgramPick(Model, Context, Rng);

        if (Token == TOKEN_EOS)
        {
            break;
        }

        Out[Length++] = Token;

        for (int i = 0; i < Model->Order - 2; i++)
        {
            Context[i] = Context[i + 1];
        }
        if (Model->Order >= 2)
        {
            Context[Model->Order - 2] = Token;
        }
    }

    return Length;
}

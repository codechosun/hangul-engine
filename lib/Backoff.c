// lib/Backoff.c

#include "Backoff.h"

#include <assert.h>
#include <string.h>

int BackoffBuild(FBackoff* Backoff, const char* Path,
                 int MaxOrder, uint64_t MaxLines)
{
    assert(Backoff != NULL);
    assert(MaxOrder >= 1 && MaxOrder <= NGRAM_MAX_ORDER);

    memset(Backoff, 0, sizeof(*Backoff));
    Backoff->MaxOrder = MaxOrder;
    Backoff->DiscountNum = BACKOFF_NUM;
    Backoff->DiscountDen = BACKOFF_DEN;

    for (int Order = 1; Order <= MaxOrder; Order++)
    {
        if (!NgramBuild(&Backoff->Orders[Order], Path, Order, MaxLines))
        {
            BackoffFree(Backoff);
            return 0;
        }
    }

    return 1;
}

void BackoffFree(FBackoff* Backoff)
{
    if (Backoff == NULL)
    {
        return;
    }

    for (int Order = 1; Order <= NGRAM_MAX_ORDER; Order++)
    {
        NgramFree(&Backoff->Orders[Order]);
    }

    memset(Backoff, 0, sizeof(*Backoff));
}

void BackoffSetDiscount(FBackoff* Backoff, uint32_t Num, uint32_t Den)
{
    assert(Backoff != NULL);
    assert(Den > 0);
    assert(Num < Den);

    Backoff->DiscountNum = Num;
    Backoff->DiscountDen = Den;
}

void BackoffResetStats(FBackoff* Backoff)
{
    assert(Backoff != NULL);

    Backoff->PickCount = 0;
    Backoff->MissSteps = 0;
    Backoff->DiscountSteps = 0;
    memset(Backoff->UsedAt, 0, sizeof(Backoff->UsedAt));
}

uint32_t BackoffPick(FBackoff* Backoff, const uint32_t* Context, int Length,
                     FRandom* Rng)
{
    assert(Backoff != NULL);
    assert(Length >= 0);

    const FNgram* Model = &Backoff->Orders[Length + 1];

    // ---- 바닥. 1그램은 물러설 곳이 없으므로 할인하지 않는다 ----
    if (Length == 0)
    {
        Backoff->PickCount++;
        Backoff->UsedAt[1]++;

        uint64_t To = Model->ContextStart[1];
        uint64_t Target = RandomBelow(Rng, Model->Cumulative[To - 1]);

        uint64_t Low = 0;
        uint64_t High = To - 1;

        while (Low < High)
        {
            uint64_t Mid = Low + (High - Low) / 2;
            if (Model->Cumulative[Mid] > Target) High = Mid;
            else                                 Low = Mid + 1;
        }

        return Model->Grams[Low];
    }

    int64_t Index = NgramFindContext(Model, Context);

    // ---- 문맥 자체를 본 적이 없다. 한 글자 줄여 다시 묻는다 ----
    if (Index < 0)
    {
        Backoff->MissSteps++;
        return BackoffPick(Backoff, Context + 1, Length - 1, Rng);
    }

    uint64_t From = Model->ContextStart[Index];
    uint64_t To = Model->ContextStart[Index + 1];
    uint64_t Choices = To - From;
    uint64_t Total = Model->Cumulative[To - 1];

    // ---- 할인 ----
    //
    // 후보 하나마다 BACKOFF_NUM/BACKOFF_DEN 만큼을 떼어 짧은 문맥에 넘긴다.
    // 전부 BACKOFF_DEN 배로 키워 정수만으로 계산한다.
    //
    //     전체    = DEN * Total
    //     넘길 몫 = NUM * Choices
    //     남는 몫 = DEN * Total - NUM * Choices
    uint64_t Scale = (uint64_t)Backoff->DiscountDen * Total;
    uint64_t Reserved = (uint64_t)Backoff->DiscountNum * Choices;

    uint64_t Draw = RandomBelow(Rng, Scale);

    if (Draw < Reserved)
    {
        Backoff->DiscountSteps++;
        return BackoffPick(Backoff, Context + 1, Length - 1, Rng);
    }

    uint64_t Target = Draw - Reserved;

    // 할인된 누적합은 DEN*Cumulative[i] - NUM*(i - From + 1) 이다.
    // 원래 값이 1 이상이므로 DEN*1 - NUM = 1 이상, 여전히 증가한다.
    uint64_t Low = From;
    uint64_t High = To - 1;

    while (Low < High)
    {
        uint64_t Mid = Low + (High - Low) / 2;
        uint64_t Cum = (uint64_t)Backoff->DiscountDen * Model->Cumulative[Mid]
                     - (uint64_t)Backoff->DiscountNum * (Mid - From + 1);

        if (Cum > Target) High = Mid;
        else              Low = Mid + 1;
    }

    Backoff->PickCount++;
    Backoff->UsedAt[Length + 1]++;

    return Model->Grams[Low * (uint64_t)Model->Order + Model->Order - 1];
}

int BackoffGenerate(FBackoff* Backoff, FRandom* Rng,
                    uint32_t* Out, int MaxLength)
{
    assert(Backoff != NULL);
    assert(Out != NULL);

    int Length = Backoff->MaxOrder - 1;

    uint32_t Context[NGRAM_MAX_ORDER];
    for (int i = 0; i < Length; i++)
    {
        Context[i] = TOKEN_BOS;
    }

    int Written = 0;

    while (Written < MaxLength)
    {
        uint32_t Token = BackoffPick(Backoff, Context, Length, Rng);

        if (Token == TOKEN_EOS)
        {
            break;
        }

        Out[Written++] = Token;

        for (int i = 0; i < Length - 1; i++)
        {
            Context[i] = Context[i + 1];
        }
        if (Length >= 1)
        {
            Context[Length - 1] = Token;
        }
    }

    return Written;
}

double BackoffProb(const FBackoff* Backoff, const uint32_t* Context, int Length,
                   uint32_t Token)
{
    assert(Backoff != NULL);
    assert(Length >= 0);

    const FNgram* Model = &Backoff->Orders[Length + 1];

    // ---- 바닥. 1그램은 할인하지 않는다 ----
    if (Length == 0)
    {
        uint64_t Total = Model->Cumulative[Model->ContextStart[1] - 1];
        uint64_t Count = NgramCount(Model, &Token);

        return (double)Count / (double)Total;
    }

    int64_t Index = NgramFindContext(Model, Context);

    // ---- 문맥 자체가 없다. 짧은 문맥이 전부를 책임진다 ----
    if (Index < 0)
    {
        return BackoffProb(Backoff, Context + 1, Length - 1, Token);
    }

    uint64_t From = Model->ContextStart[Index];
    uint64_t To = Model->ContextStart[Index + 1];
    uint64_t Choices = To - From;
    uint64_t Total = Model->Cumulative[To - 1];

    // 그램 하나를 만들어 횟수를 찾는다.
    uint32_t Gram[NGRAM_MAX_ORDER];
    for (int i = 0; i < Length; i++)
    {
        Gram[i] = Context[i];
    }
    Gram[Length] = Token;

    uint64_t Count = NgramCount(Model, Gram);

    double Scale = (double)Backoff->DiscountDen * (double)Total;

    // 이 차수가 직접 주는 몫. BackoffPick 이 뽑는 칸의 너비와 같다.
    double Direct = 0.0;
    if (Count > 0)
    {
        Direct = ((double)Backoff->DiscountDen * (double)Count
                  - (double)Backoff->DiscountNum) / Scale;
    }

    // 짧은 문맥에 넘긴 몫.
    double Reserved = (double)Backoff->DiscountNum * (double)Choices / Scale;

    return Direct + Reserved * BackoffProb(Backoff, Context + 1, Length - 1, Token);
}

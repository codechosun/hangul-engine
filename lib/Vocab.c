// lib/Vocab.c

#include "Vocab.h"
#include "Map.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB_MAX_LOAD 0.7

uint64_t VocabHash(const char* Text)
{
    // 자료구조 교재 1.7 의 65599 해시와 같은 모양이다.
    //     HashValue = 65599 * HashValue + InKey[i];
    // 상수만 64비트에 맞게 키우고, 마지막에 A4 의 MapHash 로 한 번 더 섞는다.
    uint64_t Hash = 0xCBF29CE484222325ull;

    for (const unsigned char* P = (const unsigned char*)Text; *P != '\0'; P++)
    {
        Hash ^= (uint64_t)*P;
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

int VocabInit(FVocab* Vocab, uint64_t InitialCapacity)
{
    assert(Vocab != NULL);

    memset(Vocab, 0, sizeof(*Vocab));

    uint64_t Capacity = RoundUpPowerOfTwo(InitialCapacity);

    Vocab->TextCapacity = Capacity * 16;
    Vocab->Text = (char*)malloc((size_t)Vocab->TextCapacity);

    Vocab->OffsetCapacity = Capacity;
    Vocab->Offsets = (uint64_t*)malloc((size_t)Capacity * sizeof(uint64_t));

    Vocab->SlotCapacity = Capacity;
    Vocab->Slots = (uint32_t*)calloc((size_t)Capacity, sizeof(uint32_t));

    if (Vocab->Text == NULL || Vocab->Offsets == NULL || Vocab->Slots == NULL)
    {
        VocabFree(Vocab);
        return 0;
    }

    return 1;
}

void VocabFree(FVocab* Vocab)
{
    if (Vocab == NULL)
    {
        return;
    }

    free(Vocab->Text);
    free(Vocab->Offsets);
    free(Vocab->Slots);

    memset(Vocab, 0, sizeof(*Vocab));
}

const char* VocabText(const FVocab* Vocab, uint32_t Id)
{
    assert(Vocab != NULL);

    if ((uint64_t)Id >= Vocab->Count)
    {
        return NULL;
    }

    return Vocab->Text + Vocab->Offsets[Id];
}

uint32_t VocabFind(const FVocab* Vocab, const char* Text)
{
    assert(Vocab != NULL);
    assert(Text != NULL);

    uint64_t Mask = Vocab->SlotCapacity - 1;
    uint64_t Slot = VocabHash(Text) & Mask;

    for (;;)
    {
        if (Vocab->Slots[Slot] == 0)
        {
            return VOCAB_NONE;
        }

        uint32_t Id = Vocab->Slots[Slot] - 1;
        if (strcmp(Vocab->Text + Vocab->Offsets[Id], Text) == 0)
        {
            return Id;
        }

        Slot = (Slot + 1) & Mask;
    }
}

static int GrowSlots(FVocab* Vocab)
{
    uint64_t NewCapacity = Vocab->SlotCapacity * 2;

    uint32_t* Slots = (uint32_t*)calloc((size_t)NewCapacity, sizeof(uint32_t));
    if (Slots == NULL)
    {
        return 0;
    }

    uint64_t Mask = NewCapacity - 1;

    // 아레나는 건드리지 않는다. 번호만 다시 꽂는다.
    for (uint64_t i = 0; i < Vocab->SlotCapacity; i++)
    {
        if (Vocab->Slots[i] == 0)
        {
            continue;
        }

        uint32_t Id = Vocab->Slots[i] - 1;
        uint64_t Slot = VocabHash(Vocab->Text + Vocab->Offsets[Id]) & Mask;

        while (Slots[Slot] != 0)
        {
            Slot = (Slot + 1) & Mask;
        }

        Slots[Slot] = Vocab->Slots[i];
    }

    free(Vocab->Slots);
    Vocab->Slots = Slots;
    Vocab->SlotCapacity = NewCapacity;

    return 1;
}

uint32_t VocabIntern(FVocab* Vocab, const char* Text)
{
    assert(Vocab != NULL);
    assert(Text != NULL);

    uint32_t Found = VocabFind(Vocab, Text);
    if (Found != VOCAB_NONE)
    {
        return Found;
    }

    if ((double)(Vocab->Count + 1) > (double)Vocab->SlotCapacity * VOCAB_MAX_LOAD)
    {
        if (!GrowSlots(Vocab))
        {
            return VOCAB_NONE;
        }
    }

    size_t Length = strlen(Text) + 1;

    while (Vocab->TextUsed + Length > Vocab->TextCapacity)
    {
        uint64_t NewCapacity = Vocab->TextCapacity * 2;
        char* Grown = (char*)realloc(Vocab->Text, (size_t)NewCapacity);
        if (Grown == NULL)
        {
            return VOCAB_NONE;
        }

        Vocab->Text = Grown;
        Vocab->TextCapacity = NewCapacity;
    }

    if (Vocab->Count == Vocab->OffsetCapacity)
    {
        uint64_t NewCapacity = Vocab->OffsetCapacity * 2;
        uint64_t* Grown = (uint64_t*)realloc(
            Vocab->Offsets, (size_t)NewCapacity * sizeof(uint64_t));

        if (Grown == NULL)
        {
            return VOCAB_NONE;
        }

        Vocab->Offsets = Grown;
        Vocab->OffsetCapacity = NewCapacity;
    }

    uint32_t Id = (uint32_t)Vocab->Count;

    Vocab->Offsets[Id] = Vocab->TextUsed;
    memcpy(Vocab->Text + Vocab->TextUsed, Text, Length);
    Vocab->TextUsed += Length;
    Vocab->Count++;

    uint64_t Mask = Vocab->SlotCapacity - 1;
    uint64_t Slot = VocabHash(Text) & Mask;
    while (Vocab->Slots[Slot] != 0)
    {
        Slot = (Slot + 1) & Mask;
    }
    Vocab->Slots[Slot] = Id + 1;

    return Id;
}

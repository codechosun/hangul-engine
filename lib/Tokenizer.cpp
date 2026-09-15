// lib/Tokenizer.cpp

#include "Tokenizer.hpp"
#include "Utf8.h"

#include <cstdio>
#include <cstring>

namespace
{

const size_t GTableSize = 0x11000;

} // namespace

void FTokenizer::BuildFromText(const char* Text, size_t Bytes, size_t Limit,
                               size_t InSpecialCount)
{
    SpecialCount = InSpecialCount;

    // 1. 센다. A2 에서 한 일이다.
    std::vector<uint64_t> Tally(GTableSize, 0);
    uint64_t Total = 0;

    size_t At = 0;
    while (At < Bytes)
    {
        uint32_t Code = 0;
        const int Used = Utf8Decode(Text + At, &Code);
        if (Used <= 0) break;

        At += (size_t)Used;
        if (Code < GTableSize)
        {
            Tally[Code]++;
            Total++;
        }
    }

    // 2. 많은 순서로 Limit 개를 고른다.
    //
    // 어휘가 수백 개라 매번 전체를 훑어도 된다. 어휘가 커지면
    // A2 의 FreqSort 처럼 정렬해야 한다.
    Codepoints.clear();

    uint64_t Covered = 0;

    for (size_t Rank = 0; Rank < Limit; Rank++)
    {
        uint64_t Best = 0;
        size_t Where = 0;

        for (size_t c = 0; c < GTableSize; c++)
        {
            if (Tally[c] > Best)
            {
                Best = Tally[c];
                Where = c;
            }
        }

        if (Best == 0) break;

        Codepoints.push_back((uint32_t)Where);
        Covered += Best;
        Tally[Where] = 0;
    }

    CoverageRatio = (Total > 0) ? ((double)Covered / (double)Total) : 0.0;

    // 3. 되돌리는 표.
    Table.assign(GTableSize, -1);
    for (size_t i = 0; i < Codepoints.size(); i++)
    {
        Table[Codepoints[i]] = (int)i;
    }
}

void FTokenizer::BuildFromCodepoints(const std::vector<uint32_t>& InCodepoints,
                                     size_t InSpecialCount)
{
    Codepoints = InCodepoints;
    SpecialCount = InSpecialCount;
    CoverageRatio = 0.0;   // 원문을 안 봤으므로 말할 수 없다

    Table.assign(GTableSize, -1);
    for (size_t i = 0; i < Codepoints.size(); i++)
    {
        if (Codepoints[i] < GTableSize)
        {
            Table[Codepoints[i]] = (int)i;
        }
    }
}

int FTokenizer::Find(uint32_t Codepoint) const
{
    if (Codepoint >= GTableSize) return -1;
    if (Table.empty()) return -1;

    return Table[Codepoint];
}

size_t FTokenizer::Encode(const char* Text, std::vector<uint32_t>& Out) const
{
    size_t Dropped = 0;
    size_t At = 0;

    while (Text[At] != '\0')
    {
        uint32_t Code = 0;
        const int Used = Utf8Decode(Text + At, &Code);
        if (Used <= 0) break;

        At += (size_t)Used;

        const int Id = Find(Code);
        if (Id < 0) { Dropped++; continue; }

        Out.push_back((uint32_t)Id);
    }

    return Dropped;
}

size_t FTokenizer::Encode(const std::string& Text,
                          std::vector<uint32_t>& Out) const
{
    return Encode(Text.c_str(), Out);
}

int FTokenizer::Decode(uint32_t Id, char* OutBuffer) const
{
    if ((size_t)Id >= Codepoints.size())
    {
        OutBuffer[0] = '\0';
        return 0;
    }

    const int Bytes = Utf8Encode(Codepoints[Id], OutBuffer);
    OutBuffer[Bytes] = '\0';

    return Bytes;
}

std::string FTokenizer::DecodeAll(const std::vector<uint32_t>& Ids) const
{
    std::string Result;
    char Buffer[8];

    for (size_t i = 0; i < Ids.size(); i++)
    {
        if ((size_t)Ids[i] >= Codepoints.size())
        {
            // 특수 토큰. 눈에 보이게 적는다.
            char Mark[32];
            snprintf(Mark, sizeof(Mark), "<%zu>",
                     (size_t)Ids[i] - Codepoints.size());
            Result += Mark;
            continue;
        }

        Decode(Ids[i], Buffer);
        Result += Buffer;
    }

    return Result;
}

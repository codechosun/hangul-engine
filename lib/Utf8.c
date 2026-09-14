// lib/Utf8.c

#include "Utf8.h"

#include <assert.h>
#include <stddef.h>

int Utf8Encode(uint32_t Code, char* OutBuffer)
{
    assert(OutBuffer != NULL);

    // 부호 확장을 피하려고 unsigned char 로 다룬다.
    unsigned char* Bytes = (unsigned char*)OutBuffer;

    // 1바이트: 0xxxxxxx
    if (Code <= 0x7Fu)
    {
        Bytes[0] = (unsigned char)Code;
        return 1;
    }

    // 2바이트: 110xxxxx 10xxxxxx
    if (Code <= 0x7FFu)
    {
        Bytes[0] = (unsigned char)(0xC0u | (Code >> 6));
        Bytes[1] = (unsigned char)(0x80u | (Code & 0x3Fu));
        return 2;
    }

    // 3바이트: 1110xxxx 10xxxxxx 10xxxxxx   <- 한글이 여기
    if (Code <= 0xFFFFu)
    {
        Bytes[0] = (unsigned char)(0xE0u | (Code >> 12));
        Bytes[1] = (unsigned char)(0x80u | ((Code >> 6) & 0x3Fu));
        Bytes[2] = (unsigned char)(0x80u | (Code & 0x3Fu));
        return 3;
    }

    // 4바이트: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    if (Code <= 0x10FFFFu)
    {
        Bytes[0] = (unsigned char)(0xF0u | (Code >> 18));
        Bytes[1] = (unsigned char)(0x80u | ((Code >> 12) & 0x3Fu));
        Bytes[2] = (unsigned char)(0x80u | ((Code >> 6) & 0x3Fu));
        Bytes[3] = (unsigned char)(0x80u | (Code & 0x3Fu));
        return 4;
    }

    return 0;   // 유니코드 범위 밖
}

int Utf8Decode(const char* Text, uint32_t* OutCode)
{
    assert(Text != NULL);
    assert(OutCode != NULL);

    const unsigned char* Bytes = (const unsigned char*)Text;
    unsigned char First = Bytes[0];

    // 1바이트: 0xxxxxxx
    if ((First & 0x80u) == 0x00u)
    {
        *OutCode = First;
        return 1;
    }

    // 2바이트: 110xxxxx
    if ((First & 0xE0u) == 0xC0u)
    {
        if ((Bytes[1] & 0xC0u) != 0x80u) return 0;

        *OutCode = ((uint32_t)(First    & 0x1Fu) << 6)
                 |  (uint32_t)(Bytes[1] & 0x3Fu);
        return 2;
    }

    // 3바이트: 1110xxxx
    if ((First & 0xF0u) == 0xE0u)
    {
        if ((Bytes[1] & 0xC0u) != 0x80u) return 0;
        if ((Bytes[2] & 0xC0u) != 0x80u) return 0;

        *OutCode = ((uint32_t)(First    & 0x0Fu) << 12)
                 | ((uint32_t)(Bytes[1] & 0x3Fu) << 6)
                 |  (uint32_t)(Bytes[2] & 0x3Fu);
        return 3;
    }

    // 4바이트: 11110xxx
    if ((First & 0xF8u) == 0xF0u)
    {
        if ((Bytes[1] & 0xC0u) != 0x80u) return 0;
        if ((Bytes[2] & 0xC0u) != 0x80u) return 0;
        if ((Bytes[3] & 0xC0u) != 0x80u) return 0;

        *OutCode = ((uint32_t)(First    & 0x07u) << 18)
                 | ((uint32_t)(Bytes[1] & 0x3Fu) << 12)
                 | ((uint32_t)(Bytes[2] & 0x3Fu) << 6)
                 |  (uint32_t)(Bytes[3] & 0x3Fu);
        return 4;
    }

    return 0;   // 선행 바이트가 아니다
}

int Utf8SequenceLength(unsigned char First)
{
    if ((First & 0x80u) == 0x00u) return 1;   // 0xxxxxxx
    if ((First & 0xE0u) == 0xC0u) return 2;   // 110xxxxx
    if ((First & 0xF0u) == 0xE0u) return 3;   // 1110xxxx
    if ((First & 0xF8u) == 0xF0u) return 4;   // 11110xxx

    return 0;   // 10xxxxxx 이거나 정의되지 않은 패턴
}

int Utf8Length(const char* Text)
{
    assert(Text != NULL);

    int Count = 0;

    while (*Text != '\0')
    {
        uint32_t Code = 0;
        int Len = Utf8Decode(Text, &Code);

        if (Len == 0)
        {
            return -1;   // 깨진 바이트열
        }

        Text += Len;     // 읽은 만큼 앞으로
        Count++;
    }

    return Count;
}

int Utf8IsHangul(uint32_t Code)
{
    return (Code >= HANGUL_FIRST && Code <= HANGUL_LAST);
}

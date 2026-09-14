// ch-A01-utf8/Main.c
//
// A1. UTF-8 과 한글 코드포인트

#include "Test.h"
#include "Utf8.h"

#include <stdio.h>
#include <string.h>

// A1-2. 부호 있는 char 로 바이트를 다루면 벌어지는 일.
static void ShowSignedCharTrap(void)
{
    char          Signed   = (char)0xEA;
    unsigned char Unsigned = (unsigned char)0xEA;

    // 같은 비트 0xEA 인데 값이 다르게 읽힌다.
    printf("  char          0xEA -> %d\n", Signed);     // -22
    printf("  unsigned char 0xEA -> %d\n", Unsigned);   // 234

    // 3바이트 선행 바이트인지 보려고 0xE0 과 비교하면?
    printf("  0xE0 이상인가?  char: %s   unsigned char: %s\n",
           (Signed   >= 0xE0) ? "예" : "아니오(!)",
           (Unsigned >= 0xE0) ? "예" : "아니오");
    printf("\n");
}

// A1-3. 글자 하나의 바이트를 그대로 보여준다.
static void DumpBytes(const char* Label, const char* Text)
{
    printf("  %s : ", Label);

    const unsigned char* Bytes = (const unsigned char*)Text;
    while (*Bytes != '\0')
    {
        printf("%02X ", *Bytes);
        Bytes++;
    }

    uint32_t Code = 0;
    Utf8Decode(Text, &Code);
    printf(" -> U+%04X\n", Code);
}

int main(void)
{
    printf("A1. UTF-8과 한글 코드포인트\n\n");

    // ---- A1-2. 부호 함정 ----
    printf("[A1-2] 부호 있는 char 의 함정\n");
    ShowSignedCharTrap();

    // ---- A1-3. 바이트 확인 ----
    printf("[A1-3] 글자별 바이트\n");
    DumpBytes("가", "가");
    DumpBytes("한", "한");
    DumpBytes("A", "A");
    printf("\n");

    // ---- A1-4. 인코딩 결과가 손으로 계산한 값과 맞는가 ----
    char Buffer[8] = { 0 };
    int Len = Utf8Encode(0xAC00u, Buffer);

    CHECK(Len == 3);
    CHECK((unsigned char)Buffer[0] == 0xEA);
    CHECK((unsigned char)Buffer[1] == 0xB0);
    CHECK((unsigned char)Buffer[2] == 0x80);

    // ---- A1-5. 바이트 수와 글자 수는 다르다 ----
    printf("[A1-5] strlen 과 Utf8Length\n");
    printf("  \"안녕\" : strlen=%zu, Utf8Length=%d\n",
           strlen("안녕"), Utf8Length("안녕"));
    printf("  \"Hi안녕\" : strlen=%zu, Utf8Length=%d\n",
           strlen("Hi안녕"), Utf8Length("Hi안녕"));
    printf("\n");

    CHECK(strlen("안녕") == 6);
    CHECK(Utf8Length("안녕") == 2);
    CHECK(Utf8Length("Hi안녕") == 4);
    CHECK(Utf8Length("") == 0);

    // ---- A1-6. 완성형 한글 11,172자 왕복 ----
    int RoundTripFail = 0;
    int NotThreeBytes = 0;
    int HangulCount   = 0;

    for (uint32_t Code = HANGUL_FIRST; Code <= HANGUL_LAST; Code++)
    {
        char     EncBuffer[8] = { 0 };
        uint32_t Decoded      = 0;

        int EncLen = Utf8Encode(Code, EncBuffer);
        int DecLen = Utf8Decode(EncBuffer, &Decoded);

        if (EncLen != 3 || DecLen != 3)
        {
            NotThreeBytes++;
        }

        if (Decoded != Code)
        {
            RoundTripFail++;
        }

        HangulCount++;
    }

    printf("[A1-6] 완성형 한글 왕복 검증\n");
    printf("  검사한 글자 수 : %d\n", HangulCount);
    printf("  3바이트가 아닌 것 : %d\n", NotThreeBytes);
    printf("  왕복이 깨진 것 : %d\n", RoundTripFail);
    printf("\n");

    CHECK(HangulCount == 11172);
    CHECK(NotThreeBytes == 0);
    CHECK(RoundTripFail == 0);

    // ---- A1-7. 깨진 바이트열은 거부해야 한다 ----
    uint32_t Dummy = 0;

    CHECK(Utf8Decode("\xEA\xB0", &Dummy) == 0);       // 3바이트인데 2개뿐
    CHECK(Utf8Decode("\xEA\x41\x80", &Dummy) == 0);   // 이어지는 바이트가 아님
    CHECK(Utf8Decode("\x80", &Dummy) == 0);           // 선행 바이트가 아님
    CHECK(Utf8Length("\xEA\xB0") == -1);

    return ReportResult();
}

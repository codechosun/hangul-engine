// common/Pretty.c

#include "Pretty.h"
#include "Utf8.h"

#include <stdio.h>

int DisplayWidth(const char* Text)
{
    if (Text == NULL)
    {
        return 0;
    }

    int Width = 0;
    const char* Cursor = Text;

    while (*Cursor != '\0')
    {
        uint32_t Code = 0;
        int Length = Utf8Decode(Cursor, &Code);

        if (Length <= 0)
        {
            // 깨진 바이트는 한 칸으로 치고 한 바이트만 넘어간다.
            Length = 1;
            Code = (unsigned char)*Cursor;
        }

        Width += (Code < 0x80u) ? 1 : 2;
        Cursor += Length;
    }

    return Width;
}

static void PrintSpaces(int Count)
{
    for (int i = 0; i < Count; i++)
    {
        printf(" ");
    }
}

void PrintPadded(const char* Text, int Width)
{
    printf("%s", Text);
    PrintSpaces(Width - DisplayWidth(Text));
}

void PrintPaddedRight(const char* Text, int Width)
{
    PrintSpaces(Width - DisplayWidth(Text));
    printf("%s", Text);
}

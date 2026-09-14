// lib/Scan.c

#include "Scan.h"
#include "Utf8.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

int ScanOpenSized(FScanner* Scanner, const char* Path, size_t Chunk)
{
    assert(Scanner != NULL);
    assert(Path != NULL);

    if (Chunk == 0)
    {
        Chunk = SCAN_CHUNK;
    }

    Scanner->File = fopen(Path, "rb");
    if (Scanner->File == NULL)
    {
        return 0;
    }

    // 뒤쪽 여유 8바이트는 청크 경계에서 잘린 글자를 담아두는 자리.
    Scanner->Buffer = (char*)malloc(Chunk + 8);
    if (Scanner->Buffer == NULL)
    {
        fclose(Scanner->File);
        Scanner->File = NULL;
        return 0;
    }

    Scanner->Chunk = Chunk;
    Scanner->Filled = 0;
    Scanner->Pos = 0;
    Scanner->Done = 0;

    return 1;
}

int ScanOpen(FScanner* Scanner, const char* Path)
{
    return ScanOpenSized(Scanner, Path, SCAN_CHUNK);
}

void ScanClose(FScanner* Scanner)
{
    if (Scanner == NULL)
    {
        return;
    }

    if (Scanner->File != NULL)
    {
        fclose(Scanner->File);
        Scanner->File = NULL;
    }

    free(Scanner->Buffer);
    Scanner->Buffer = NULL;
    Scanner->Filled = 0;
    Scanner->Pos = 0;
    Scanner->Done = 1;
}

// 못 읽고 남은 꼬리를 앞으로 당기고 그 뒤를 새로 채운다.
static void Refill(FScanner* Scanner)
{
    size_t Leftover = Scanner->Filled - Scanner->Pos;
    memmove(Scanner->Buffer, Scanner->Buffer + Scanner->Pos, Leftover);

    size_t Read = fread(Scanner->Buffer + Leftover, 1, Scanner->Chunk, Scanner->File);

    Scanner->Filled = Leftover + Read;
    Scanner->Pos = 0;

    if (Read == 0)
    {
        Scanner->Done = 1;   // 파일 끝. 남은 꼬리는 깨진 바이트다.
    }
}

int ScanNext(FScanner* Scanner, uint32_t* OutCode)
{
    assert(Scanner != NULL);
    assert(OutCode != NULL);

    for (;;)
    {
        if (Scanner->Pos >= Scanner->Filled)
        {
            if (Scanner->Done)
            {
                return 0;
            }
            Refill(Scanner);
            continue;
        }

        unsigned char First = (unsigned char)Scanner->Buffer[Scanner->Pos];
        int Need = Utf8SequenceLength(First);

        if (Need == 0)
        {
            Scanner->Pos++;      // 선행 바이트가 아니다. 한 바이트 버린다.
            continue;
        }

        if (Scanner->Pos + (size_t)Need > Scanner->Filled)
        {
            if (Scanner->Done)
            {
                return 0;        // 끝에서 잘렸다. 버린다.
            }
            Refill(Scanner);
            continue;
        }

        uint32_t Code = 0;
        int Got = Utf8Decode(Scanner->Buffer + Scanner->Pos, &Code);
        Scanner->Pos += (size_t)Need;

        if (Got == Need && Code < CODEPOINT_LIMIT)
        {
            *OutCode = Code;
            return 1;
        }

        // 깨진 바이트열이다. 건너뛰고 다음 글자를 찾는다.
    }
}

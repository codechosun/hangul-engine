// lib/Scan.h
//
// 텍스트 파일을 청크 단위로 읽으며 코드포인트를 하나씩 꺼내준다.
//
// A2 에서 lib/Freq.c 안에 손으로 써 넣었던 루프다.
// A4 에서 같은 루프가 두 번째로 필요해져서 밖으로 꺼냈다.
//
// 청크 경계에서 글자가 잘리는 문제를 여기서 처리하므로,
// 쓰는 쪽은 "다음 글자 주세요" 만 반복하면 된다.

#ifndef SCAN_H
#define SCAN_H

#include <stdint.h>
#include <stdio.h>

// 한 번에 읽어들일 크기. 뒤쪽 여유 8바이트는 잘린 글자를 담아두는 자리.
#define SCAN_CHUNK (1u << 20)

typedef struct
{
    FILE*  File;
    char*  Buffer;
    size_t Filled;     // Buffer 에 들어 있는 바이트 수
    size_t Pos;        // 다음에 읽을 자리
    int    Done;       // 파일을 끝까지 읽었는가
} FScanner;

// 성공하면 1, 파일을 못 열거나 메모리가 모자라면 0.
int ScanOpen(FScanner* Scanner, const char* Path);

void ScanClose(FScanner* Scanner);

// 다음 코드포인트를 OutCode 에 담고 1 을 준다. 더 없으면 0.
int ScanNext(FScanner* Scanner, uint32_t* OutCode);

#endif

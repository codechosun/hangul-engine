// lib/Utf8.h
//
// UTF-8 바이트열과 유니코드 코드포인트 사이를 오가는 최소 도구.

#ifndef UTF8_H
#define UTF8_H

#include <stdint.h>

// 완성형 한글의 코드포인트 범위. 총 11,172자.
#define HANGUL_FIRST 0xAC00u
#define HANGUL_LAST  0xD7A3u

// 코드포인트 하나를 UTF-8 바이트열로 쓴다.
// OutBuffer 는 최소 4바이트여야 한다.
// 쓴 바이트 수를 돌려준다. 코드포인트가 범위 밖이면 0.
int Utf8Encode(uint32_t Code, char* OutBuffer);

// UTF-8 바이트열에서 코드포인트 하나를 읽는다.
// 읽은 바이트 수를 돌려준다. 잘못된 바이트열이면 0.
int Utf8Decode(const char* Text, uint32_t* OutCode);

// 바이트 수가 아니라 글자 수를 센다.
// 잘못된 바이트열을 만나면 -1.
int Utf8Length(const char* Text);

// 코드포인트가 완성형 한글인지 본다.
int Utf8IsHangul(uint32_t Code);

#endif

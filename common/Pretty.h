// common/Pretty.h
//
// 표를 화면에 반듯하게 찍기 위한 도구. 엔진에는 들어가지 않는다.
//
// printf 의 "%-18s" 는 **바이트 수**로 칸을 맞춘다.
// 한글 한 글자는 UTF-8 로 3바이트이면서 화면에서는 2칸을 차지하므로,
// 한글이 섞인 표는 반드시 어긋난다.
//
// A3 에서는 폭이 들쭉날쭉한 칸을 맨 뒤로 보내서 피했다.
// A4 부터는 표가 여러 개라 피할 수가 없어서 직접 센다.

#ifndef PRETTY_H
#define PRETTY_H

#ifdef __cplusplus
extern "C" {
#endif

// 이 문자열이 화면에서 차지하는 칸 수.
// ASCII 는 1칸, 그 밖은 2칸으로 친다. 한글·한자·전각 기호가 모두 2칸이므로
// 이 교재가 찍는 범위에서는 이 어림이 정확하다.
int DisplayWidth(const char* Text);

// 왼쪽 정렬로 Width 칸을 채워 찍는다.
void PrintPadded(const char* Text, int Width);

// 오른쪽 정렬.
void PrintPaddedRight(const char* Text, int Width);

#ifdef __cplusplus
}
#endif
#endif

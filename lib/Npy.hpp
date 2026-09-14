// lib/Npy.hpp
//
// NumPy 의 .npy 파일을 읽고 쓴다.
//
// A7 에서 우리가 직접 설계한 HGNG 형식과 판박이다.
//
//     매직 넘버 + 버전 + 헤더 + 날것 데이터
//
// 다른 것은 **남이 정한 형식**이라는 것뿐이다. 우리가 고칠 수 없고,
// 헤더도 우리 마음대로 못 읽는다. 그쪽이 정한 대로 파싱해야 한다.
//
// 구조 (버전 1.0)
//
//     0   \x93NUMPY                6바이트
//     6   메이저, 마이너            2바이트  (1, 0)
//     8   헤더 길이 (리틀 엔디언)   2바이트
//    10   헤더 (아스키 문자열)      헤더 길이만큼
//         {'descr': '<f4', 'fortran_order': False, 'shape': (512, 16), }
//         뒤를 공백으로 채우고 줄바꿈으로 끝낸다.
//         전체 길이(10 + 헤더 길이)가 64의 배수가 되게 맞춘다.
//    ...  날것 데이터. 행 우선으로 쭉 이어진다

#ifndef NPY_HPP
#define NPY_HPP

#include "Matrix.hpp"
#include "Types.h"

#include <cstddef>
#include <string>
#include <vector>

// 파일이 담고 있는 것.
struct FNpyInfo
{
    std::string Descr;        // "<f4" 또는 "<f8"
    bool bFortranOrder = false;
    size_t Rows = 0;
    size_t Cols = 0;          // 1차원이면 0
    size_t ElementSize = 0;   // 4 또는 8
    size_t HeaderLength = 0;
};

// 헤더만 읽는다. 성공하면 1.
int NpyReadInfo(const char* Path, FNpyInfo& Info);

// 2차원 실수 배열을 쓴다. Real 의 크기에 맞춰 '<f4' 또는 '<f8' 로 적는다.
int NpySaveMatrix(const char* Path, const FMatrix& M);

// 2차원 실수 배열을 읽는다.
//
// 파일이 float32 인데 Real 이 double 이거나 그 반대여도 괜찮다.
// 읽으면서 변환한다.
int NpyLoadMatrix(const char* Path, FMatrix& Out);

#endif

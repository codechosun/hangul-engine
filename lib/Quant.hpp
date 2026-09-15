// lib/Quant.hpp
//
// 양자화. 실수 가중치를 작은 정수로 바꿔 담는다.
//
//     양자화   q = round(x / scale)
//     복원     x ~= q * scale
//
// 8비트면 크기가 4분의 1, 4비트면 8분의 1이 된다.
// 대신 값이 조금 틀어진다. **얼마나 틀어지는지를 재는 것**이 D5 의 일이다.
//
// scale 을 어디 단위로 둘 것인가가 정확도를 좌우한다.
//
//     행렬 전체에 하나   간단하다. 값의 폭이 크면 손해가 크다
//     줄마다 하나        조금 복잡하다. 훨씬 정확하다
//
// A1 이후 처음으로 비트 연산(C 교재 2.4)이 다시 나온다. 4비트를 담으려면
// 한 바이트에 둘을 욱여넣고 시프트와 마스크로 꺼내야 한다.

#ifndef QUANT_HPP
#define QUANT_HPP

#include "Tensor.hpp"
#include "Types.h"

#include <cstdint>
#include <vector>

// 8비트로 담은 2차원 가중치. (행, 열)
struct FQuant8
{
    std::vector<int8_t> Values;   // 행 * 열
    std::vector<Real> Scales;     // 행마다 하나. 전체 하나면 크기 1
    size_t Rows = 0;
    size_t Cols = 0;
    bool bPerRow = true;

    size_t Bytes() const;
};

// 4비트. 한 바이트에 두 개가 들어간다.
struct FQuant4
{
    std::vector<uint8_t> Packed;  // (행 * 열 + 1) / 2
    std::vector<Real> Scales;
    size_t Rows = 0;
    size_t Cols = 0;

    size_t Bytes() const;
};

// 2차원 텐서를 8비트로 담는다.
FQuant8 Quantize8(const FTensor& T, bool bPerRow);

// 다시 실수로 펼친다. 원본과 정확히 같지는 않다.
FTensor Dequantize8(const FQuant8& Q);

FQuant4 Quantize4(const FTensor& T);
FTensor Dequantize4(const FQuant4& Q);

// 양자화된 행렬에 실수를 곱한다. 펼치지 않고 바로 계산한다.
//
// X 는 (..., 행) 이고 결과는 (..., 열) 이다. FDense 와 같은 규약이다.
//
// 고리 순서는 C5 의 i-k-j 를 그대로 쓴다. 줄마다 다른 scale 이 **k 축**에
// 붙어 있으므로 안쪽 고리 밖으로 빼낼 수 있다. 줄 단위 scale 이 공짜인
// 이유가 이것이다. 순서를 바꾸면 공짜가 아니게 된다.
FTensor QuantizedForward8(const FQuant8& Q, const FTensor& X);

// 일부러 내적 순서(i-o-k)로 쓴 판. D5 본문에서 재보기 위한 것이다.
// 결과는 같고 속도만 다르다.
FTensor QuantizedForward8Dot(const FQuant8& Q, const FTensor& X);

// 두 텐서가 얼마나 다른가.
struct FError
{
    double MaxAbsolute = 0.0;
    double MeanAbsolute = 0.0;
    double RelativeRms = 0.0;   // 오차의 RMS / 원본의 RMS
};

FError CompareTensors(const FTensor& A, const FTensor& B);

#endif

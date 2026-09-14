// lib/Npy.cpp

#include "Npy.hpp"

#include <cstdio>
#include <cstring>

namespace
{

const unsigned char GMagic[6] = { 0x93u, 'N', 'U', 'M', 'P', 'Y' };

// 헤더 문자열에서 "'키':" 뒤의 값을 잘라 온다.
//
// 제대로 하려면 파이썬 dict 파서를 써야 한다. 그런데 이 형식이 담는
// 키는 셋뿐이고 순서도 사실상 고정이라, 문자열 자르기로 충분하다.
// **필요한 만큼만 파싱한다** 는 것도 판단이다.
std::string ValueAfter(const std::string& Header, const char* Key)
{
    std::string Needle = "'";
    Needle += Key;
    Needle += "':";

    size_t At = Header.find(Needle);
    if (At == std::string::npos)
    {
        return std::string();
    }

    At += Needle.size();
    while (At < Header.size() && Header[At] == ' ')
    {
        At++;
    }

    // shape 는 괄호 안에 쉼표가 있으므로 괄호 깊이를 센다.
    size_t Stop = At;
    int Depth = 0;
    while (Stop < Header.size())
    {
        char C = Header[Stop];
        if (C == '(') Depth++;
        if (C == ')') Depth--;
        if (C == ',' && Depth == 0) break;
        Stop++;
    }

    return Header.substr(At, Stop - At);
}

std::string StripQuotes(const std::string& Text)
{
    std::string Result;
    for (size_t i = 0; i < Text.size(); i++)
    {
        char C = Text[i];
        if (C != '\'' && C != '"' && C != ' ')
        {
            Result += C;
        }
    }
    return Result;
}

} // namespace

int NpyReadInfo(const char* Path, FNpyInfo& Info)
{
    FILE* File = fopen(Path, "rb");
    if (File == NULL)
    {
        return 0;
    }

    unsigned char Magic[6] = {};
    unsigned char Version[2] = {};
    unsigned char LengthBytes[2] = {};

    int Ok = 1;
    Ok = Ok && (fread(Magic, 1, 6, File) == 6);
    Ok = Ok && (memcmp(Magic, GMagic, 6) == 0);
    Ok = Ok && (fread(Version, 1, 2, File) == 2);
    Ok = Ok && (Version[0] == 1);
    Ok = Ok && (fread(LengthBytes, 1, 2, File) == 2);

    if (!Ok)
    {
        fclose(File);
        return 0;
    }

    // 헤더 길이는 **리틀 엔디언 2바이트**라고 형식이 못박고 있다.
    // 우리 기계가 무엇이든 이렇게 조립해야 맞다.
    size_t HeaderLength = (size_t)LengthBytes[0] | ((size_t)LengthBytes[1] << 8);

    std::vector<char> Header(HeaderLength + 1, 0);
    if (fread(Header.data(), 1, HeaderLength, File) != HeaderLength)
    {
        fclose(File);
        return 0;
    }

    fclose(File);

    std::string Text(Header.data(), HeaderLength);

    Info.HeaderLength = HeaderLength;
    Info.Descr = StripQuotes(ValueAfter(Text, "descr"));

    std::string Order = StripQuotes(ValueAfter(Text, "fortran_order"));
    Info.bFortranOrder = (Order == "True");

    std::string Shape = ValueAfter(Text, "shape");

    Info.Rows = 0;
    Info.Cols = 0;
    {
        size_t Numbers[2] = { 0, 0 };
        int Found = 0;
        size_t i = 0;

        while (i < Shape.size() && Found < 2)
        {
            if (Shape[i] >= '0' && Shape[i] <= '9')
            {
                size_t Value = 0;
                while (i < Shape.size() && Shape[i] >= '0' && Shape[i] <= '9')
                {
                    Value = Value * 10 + (size_t)(Shape[i] - '0');
                    i++;
                }
                Numbers[Found++] = Value;
            }
            else
            {
                i++;
            }
        }

        if (Found == 1)
        {
            Info.Rows = Numbers[0];
            Info.Cols = 0;
        }
        else if (Found == 2)
        {
            Info.Rows = Numbers[0];
            Info.Cols = Numbers[1];
        }
    }

    if (Info.Descr == "<f4")      Info.ElementSize = 4;
    else if (Info.Descr == "<f8") Info.ElementSize = 8;
    else                          Info.ElementSize = 0;

    return (Info.ElementSize > 0) ? 1 : 0;
}

int NpySaveMatrix(const char* Path, const FMatrix& M)
{
    FILE* File = fopen(Path, "wb");
    if (File == NULL)
    {
        return 0;
    }

    const char* Descr = (sizeof(Real) == 8) ? "<f8" : "<f4";

    char Body[128];
    int BodyLength = snprintf(
        Body, sizeof(Body),
        "{'descr': '%s', 'fortran_order': False, 'shape': (%zu, %zu), }",
        Descr, M.Rows(), M.Cols());

    // 전체 길이(10 + 헤더)가 64의 배수여야 한다. 데이터가 64바이트 경계에
    // 놓이게 하려는 것이다. 이런 배려가 형식에 박혀 있다는 것 자체가
    // 성능을 염두에 둔 설계라는 뜻이다.
    size_t Total = 10 + (size_t)BodyLength + 1;   // 끝의 줄바꿈 한 칸
    size_t Padded = ((Total + 63) / 64) * 64;
    size_t HeaderLength = Padded - 10;

    std::vector<char> Header(HeaderLength, ' ');
    memcpy(Header.data(), Body, (size_t)BodyLength);
    Header[HeaderLength - 1] = '\n';

    unsigned char LengthBytes[2];
    LengthBytes[0] = (unsigned char)(HeaderLength & 0xFF);
    LengthBytes[1] = (unsigned char)((HeaderLength >> 8) & 0xFF);

    unsigned char Version[2] = { 1, 0 };

    int Ok = 1;
    Ok = Ok && (fwrite(GMagic, 1, 6, File) == 6);
    Ok = Ok && (fwrite(Version, 1, 2, File) == 2);
    Ok = Ok && (fwrite(LengthBytes, 1, 2, File) == 2);
    Ok = Ok && (fwrite(Header.data(), 1, HeaderLength, File) == HeaderLength);
    Ok = Ok && (fwrite(M.Data(), sizeof(Real), M.Count(), File) == M.Count());

    fclose(File);
    return Ok;
}

int NpyLoadMatrix(const char* Path, FMatrix& Out)
{
    FNpyInfo Info;
    if (!NpyReadInfo(Path, Info))
    {
        return 0;
    }

    if (Info.bFortranOrder)
    {
        return 0;   // 열 우선은 안 읽는다. 읽으려면 옮겨 담아야 한다
    }

    size_t Rows = Info.Rows;
    size_t Cols = (Info.Cols == 0) ? 1 : Info.Cols;

    FILE* File = fopen(Path, "rb");
    if (File == NULL)
    {
        return 0;
    }

    if (fseek(File, (long)(10 + Info.HeaderLength), SEEK_SET) != 0)
    {
        fclose(File);
        return 0;
    }

    FMatrix Result(Rows, Cols);
    size_t Count = Rows * Cols;

    int Ok = 1;

    if (Info.ElementSize == sizeof(Real))
    {
        // 크기가 같으면 통째로 읽는다.
        Ok = (fread(Result.Data(), sizeof(Real), Count, File) == Count);
    }
    else if (Info.ElementSize == 4)
    {
        std::vector<float> Buffer(Count);
        Ok = (fread(Buffer.data(), 4, Count, File) == Count);
        for (size_t i = 0; i < Count && Ok; i++)
        {
            Result.Data()[i] = (Real)Buffer[i];
        }
    }
    else
    {
        std::vector<double> Buffer(Count);
        Ok = (fread(Buffer.data(), 8, Count, File) == Count);
        for (size_t i = 0; i < Count && Ok; i++)
        {
            Result.Data()[i] = (Real)Buffer[i];
        }
    }

    fclose(File);

    if (!Ok)
    {
        return 0;
    }

    Out = Result;
    return 1;
}

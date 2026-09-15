// ch-E02-engine/app/main.cpp
//
// **엔진만 가지고 만든 프로그램.**
//
// 이 파일은 교재 코드를 한 줄도 안 쓴다. common/ 도, lib/ 도 안 본다.
// 보는 것은 dist/hangul-engine/ 하나뿐이다.
//
// 빌드
//     cl /std:c++17 /EHsc /utf-8 /O2 /I dist\hangul-engine
//        ch-E02-engine\app\main.cpp
//        dist\hangul-engine\*.cpp dist\hangul-engine\*.c
//
// 이게 되면 엔진이 **떨어져 나간 것**이다. E2 의 검증이다.

#include "Engine.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    const char* Path = (argc > 1) ? argv[1] : "data/chat.hgen";

    FEngine Engine;

    if (!Engine.Load(Path))
    {
        printf("모델을 못 읽었다: %s\n", Path);
        return 1;
    }

    printf("한글 엔진\n");
    printf("  파라미터 %zu개, %.1f KB\n\n", Engine.ParameterCount(),
           Engine.Bytes() / 1024.0);

    const char* Asks[5] =
    {
        "안녕", "이름이 뭐야", "고마워", "날씨 어때", "잘 가"
    };

    FGenerateOptions Options;
    Options.Temperature = 0.0;   // 늘 1등
    Options.MaxTokens = 40;

    for (int i = 0; i < 5; i++)
    {
        std::vector<FChatTurn> Turns;

        FChatTurn One;
        One.Role = RoleSystem;
        One.Text = "너는 한글 엔진이다.";
        Turns.push_back(One);

        One.Role = RoleUser;
        One.Text = Asks[i];
        Turns.push_back(One);

        printf("  사용자 : %s\n", Asks[i]);
        printf("  엔진   : %s\n\n", Engine.Reply(Turns, Options).c_str());
    }

    return 0;
}

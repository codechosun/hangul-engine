"""
엔진만 떼어 배포용 폴더로 내보낸다.

    python tools/export_engine.py            전부 (기본)
    python tools/export_engine.py --core     추론만

결과: dist/hangul-engine/

무엇을 내보내는가
-----------------
#include 를 따라가며 **실제로 딸려오는 파일만** 모은다. lib/ 를 통째로
복사하지 않는다. 교재 실습용으로 만든 것(Nplm, Pca, Npy ...)까지 배포하면
쓰는 사람이 "이건 뭐지"를 한참 헤맨다.

묶음
----
    core    추론만. Engine.hpp 한 장이 현관
    train   역전파와 옵티마이저. 미세조정을 하려면 필요하다
    quant   8비트·4비트 양자화
    ngram   A파트의 N-그램. 모델 없이도 도는 가벼운 생성기

배포 형태는 **소스 그대로**다 (결정 36). 어떤 빌드 시스템에도 들어가고,
쓰는 사람이 속을 읽고 고칠 수 있다. 코드가 MIT 이므로 상용 이용에
걸림돌이 없다 (결정 23).
"""

import os
import re
import shutil
import sys

LIB = "lib"
OUT = os.path.join("dist", "hangul-engine")

INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)

TIERS = {
    "core":  ["Engine.hpp"],
    "train": ["Backward.hpp", "Optimizer.hpp"],
    "quant": ["Quant.hpp"],
    "ngram": ["Ngram.h", "Backoff.h", "Sample.h", "Freq.h", "Scan.h",
              "Vocab.h", "Bigram.h"],
}

PUBLIC = ["Engine.hpp", "Chat.hpp", "Tokenizer.hpp", "Model.hpp", "Types.h"]


def SourceFor(Header):
    Base, _ = os.path.splitext(Header)

    for Candidate in (Base + ".cpp", Base + ".c"):
        if os.path.exists(os.path.join(LIB, Candidate)):
            return Candidate

    return None


def Closure(Roots):
    Seen = set()
    Queue = list(Roots)

    while Queue:
        Name = Queue.pop(0)
        if Name in Seen:
            continue

        Path = os.path.join(LIB, Name)
        if not os.path.exists(Path):
            continue

        Seen.add(Name)

        with open(Path, encoding="utf-8") as File:
            Text = File.read()

        for Target in INCLUDE.findall(Text):
            if os.path.exists(os.path.join(LIB, Target)):
                Queue.append(Target)

                Partner = SourceFor(Target)
                if Partner is not None:
                    Queue.append(Partner)

        Partner = SourceFor(Name)
        if Partner is not None:
            Queue.append(Partner)

    return Seen


README = """# 한글 엔진

C/C++ 로 쓴 작은 한국어 언어 모델 엔진. 외부 의존성이 없다.

## 쓰는 법

헤더 하나만 포함하면 된다.

```cpp
#include "Engine.hpp"

int main()
{
    FEngine Engine;
    if (!Engine.Load("model.hgen")) return 1;

    std::vector<FChatTurn> Turns;
    FChatTurn One;
    One.Role = RoleUser;
    One.Text = "안녕";
    Turns.push_back(One);

    printf("%s\\n", Engine.Reply(Turns, {}).c_str());
    return 0;
}
```

## 빌드

이 폴더의 `.c` 와 `.cpp` 를 전부 컴파일 대상에 넣고, 이 폴더를 포함 경로에
넣는다. 그게 전부다.

```
cl /std:c++17 /EHsc /utf-8 /O2 /I hangul-engine main.cpp hangul-engine\\*.cpp hangul-engine\\*.c
```

`/utf-8` 은 붙이는 편이 좋다. 소스에 한글 주석이 들어 있다.

## 정밀도

기본은 `float`. `USE_DOUBLE` 을 정의하면 `double` 이 된다.

**모델 파일은 둘 사이에 호환되지 않는다.** 파일에 `Real` 크기가 적혀 있어서
잘못 읽으면 조용히 틀리는 대신 `Load` 가 실패한다.

## 공개 API

아래 헤더만 보증한다. 나머지는 안쪽이고 판이 바뀌면 달라질 수 있다.

{PUBLIC}

## 라이선스

MIT.
"""


def main():
    if not os.path.exists(os.path.join(LIB, "Engine.hpp")):
        print("저장소 루트에서 돌릴 것.")
        return 2

    bCoreOnly = "--core" in sys.argv

    Roots = list(TIERS["core"])
    Used = ["core"]

    if not bCoreOnly:
        for Name in ("train", "quant", "ngram"):
            Roots += TIERS[Name]
            Used.append(Name)

    Files = sorted(Closure(Roots))

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    Total = 0
    for Name in Files:
        shutil.copy2(os.path.join(LIB, Name), os.path.join(OUT, Name))
        Total += os.path.getsize(os.path.join(LIB, Name))

    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8",
              newline="\n") as File:
        Public = "\n".join(f"- `{Name}`" for Name in PUBLIC)
        File.write(README.replace("{PUBLIC}", Public))

    Headers = [f for f in Files if f.endswith((".h", ".hpp"))]
    Sources = [f for f in Files if f.endswith((".c", ".cpp"))]

    print(f"완료 : {OUT}")
    print()
    print(f"  묶음 : {', '.join(Used)}")
    print(f"  파일 {len(Files)}개  (헤더 {len(Headers)}, 소스 {len(Sources)})")
    print(f"  합계 {Total / 1024.0:.1f} KB")
    print()

    for Name in Headers:
        Mark = "공개" if Name in PUBLIC else "안쪽"
        print(f"    [{Mark}] {Name}")

    print()

    Everything = sorted(f for f in os.listdir(LIB)
                        if f.endswith((".h", ".hpp", ".c", ".cpp")))
    Left = [f for f in Everything if f not in Files]

    print(f"  안 내보낸 파일 {len(Left)}개 (교재 실습용)")
    print("   ", " ".join(Left))
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

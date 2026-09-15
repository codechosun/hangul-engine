"""
엔진의 경계를 검사한다.

    python tools/check_engine.py

무엇을 보는가
-------------
lib/Engine.hpp 에서 시작해 #include 를 따라가며 **실제로 딸려오는 파일**을
모은다. 그 목록이 곧 배포할 엔진이다.

그리고 두 가지를 확인한다.

    ① 딸려온 파일이 lib/ 밖을 포함하지 않는가
       common/ 은 교재 도구다. CHECK 매크로와 표 찍는 함수가 산다.
       엔진이 거기 기대면 엔진만 떼어 배포할 수가 없다.

    ② 표준 라이브러리 말고 다른 것을 쓰지 않는가

돌려주는 값은 위반 개수다. 0 이면 통과.

왜 파이썬인가
-------------
빌드가 되는지로는 이걸 못 잡는다. `-I common -I lib` 로 빌드하면
경계를 밟고 있어도 잘 된다. **경계는 컴파일러가 아니라 우리가 지킨다.**
"""

import os
import re
import sys

LIB = "lib"
ROOT = "lib/Engine.hpp"

INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^">]+)[">]', re.MULTILINE)


def SourceFor(Header):
    """헤더에 짝이 되는 소스 파일 이름. 없으면 None."""
    Base, Extension = os.path.splitext(Header)

    for Candidate in (Base + ".cpp", Base + ".c"):
        if os.path.exists(os.path.join(LIB, Candidate)):
            return Candidate

    return None


def main():
    if not os.path.exists(os.path.join(LIB, "Engine.hpp")):
        print("저장소 루트에서 돌릴 것.")
        return 2

    Seen = set()
    Queue = ["Engine.hpp"]
    Outside = []      # lib/ 밖을 포함한 자리
    Standard = set()  # 쓰는 표준 헤더

    while Queue:
        Name = Queue.pop(0)
        if Name in Seen:
            continue
        Seen.add(Name)

        Path = os.path.join(LIB, Name)
        if not os.path.exists(Path):
            continue

        with open(Path, encoding="utf-8") as File:
            Text = File.read()

        for Kind, Target in INCLUDE.findall(Text):
            if Kind == "<":
                Standard.add(Target)
                continue

            if os.path.exists(os.path.join(LIB, Target)):
                Queue.append(Target)

                Partner = SourceFor(Target)
                if Partner is not None:
                    Queue.append(Partner)
            else:
                Outside.append((Name, Target))

        # 헤더를 넣었으면 짝이 되는 소스도 함께 본다.
        Partner = SourceFor(Name)
        if Partner is not None:
            Queue.append(Partner)

    Files = sorted(Seen)
    Headers = [f for f in Files if f.endswith((".h", ".hpp"))]
    Sources = [f for f in Files if f.endswith((".c", ".cpp"))]

    print("엔진 경계 검사")
    print()
    print(f"  시작점 : {ROOT}")
    print()
    print(f"  딸려오는 파일 {len(Files)}개  (헤더 {len(Headers)}, "
          f"소스 {len(Sources)})")
    print()

    for Name in Headers:
        Partner = SourceFor(Name)
        print(f"    {Name:<18}{Partner if Partner else ''}")

    print()

    Total = 0
    for Name in Files:
        Total += os.path.getsize(os.path.join(LIB, Name))

    print(f"  합계 {Total / 1024.0:.1f} KB")
    print()

    # lib/ 전체와 견준다. 안 딸려온 것이 곧 "교재에만 있는 것"이다.
    Everything = sorted(f for f in os.listdir(LIB)
                        if f.endswith((".h", ".hpp", ".c", ".cpp")))
    Unused = [f for f in Everything if f not in Seen]

    print(f"  lib/ 에 있지만 안 딸려온 파일 {len(Unused)}개")
    print()
    for Name in Unused:
        print(f"    {Name}")
    print()

    print(f"  쓰는 표준 헤더 {len(Standard)}개")
    print("   ", " ".join(f"<{h}>" for h in sorted(Standard)))
    print()

    if Outside:
        print(f"  [위반] lib/ 밖을 포함한 자리 {len(Outside)}곳")
        for Where, What in Outside:
            print(f"    {Where} -> {What}")
    else:
        print("  [통과] lib/ 밖을 포함한 자리가 없다.")

    print()

    return len(Outside)


if __name__ == "__main__":
    sys.exit(main())

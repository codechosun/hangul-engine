"""
C4 가 읽을 .npy 파일을 만들고, C++ 이 쓴 .npy 를 읽어 확인한다.

    python tools/make_npy_expected.py          # 파이썬이 쓴다
    python tools/make_npy_expected.py --check  # C++ 이 쓴 것을 읽는다

만드는 것
---------
    data/npy_from_python.npy    파이썬이 쓴 float32 2차원 배열
    data/npy_values.csv         같은 값을 사람이 읽을 수 있게

확인하는 것 (--check)
---------------------
    data/npy_from_cpp.npy       C++ 이 쓴 것. 값이 같은지 본다
    data/npy_embedding.npy      C3 의 임베딩. 모양만 확인한다
"""

import os
import sys

import numpy as np

FROM_PYTHON = os.path.join("data", "npy_from_python.npy")
FROM_CPP = os.path.join("data", "npy_from_cpp.npy")
EMBEDDING = os.path.join("data", "npy_embedding.npy")
VALUES = os.path.join("data", "npy_values.csv")

ROWS = 5
COLS = 7


def Make():
    # 정수식으로 정해지는 값. 두 언어에서 같은 비트가 나온다.
    Values = np.empty((ROWS, COLS), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            i = r * COLS + c
            Values[r, c] = np.float32(((i * 37 + 13) % 53 - 26) / 32.0)

    os.makedirs("data", exist_ok=True)
    np.save(FROM_PYTHON, Values)

    with open(VALUES, "w", encoding="utf-8", newline="\n") as Out:
        Out.write("Row,Col,Value\n")
        for r in range(ROWS):
            for c in range(COLS):
                Out.write(f"{r},{c},{float(Values[r, c]):.17g}\n")

    Size = os.path.getsize(FROM_PYTHON)
    print(f"완료 : {FROM_PYTHON}  {Size} 바이트")
    print(f"완료 : {VALUES}")
    print()
    print(f"  모양 {Values.shape}, 타입 {Values.dtype}")
    print(f"  헤더 = {open(FROM_PYTHON, 'rb').read(128)[10:74]!r}")
    print()
    print("  값")
    for r in range(ROWS):
        print("   ", " ".join(f"{float(V):8.5f}" for V in Values[r]))
    return 0


def Check():
    if not os.path.exists(FROM_CPP):
        print(f"{FROM_CPP} 가 없다. C4 예제를 먼저 돌릴 것.")
        return 1

    Mine = np.load(FROM_PYTHON)
    Theirs = np.load(FROM_CPP)

    print(f"파이썬이 쓴 것 : 모양 {Mine.shape}, 타입 {Mine.dtype}")
    print(f"C++ 이 쓴 것   : 모양 {Theirs.shape}, 타입 {Theirs.dtype}")

    if Mine.shape != Theirs.shape:
        print("모양이 다르다.")
        return 1

    Same = np.array_equal(Mine.astype(np.float64), Theirs.astype(np.float64))
    Worst = float(np.max(np.abs(Mine.astype(np.float64)
                                - Theirs.astype(np.float64))))

    print(f"  값이 정확히 같은가 = {Same}")
    print(f"  최대 차이          = {Worst:.3e}")

    if os.path.exists(EMBEDDING):
        Embedding = np.load(EMBEDDING)
        print()
        print(f"C3 임베딩      : 모양 {Embedding.shape}, 타입 {Embedding.dtype}")
        print(f"  평균 {float(Embedding.mean()):+.6f}, "
              f"표준편차 {float(Embedding.std()):.6f}")
        print(f"  최소 {float(Embedding.min()):+.6f}, "
              f"최대 {float(Embedding.max()):+.6f}")

    return 0 if Same else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        return Check()
    return Make()


if __name__ == "__main__":
    sys.exit(main())

"""
아주 작은 NPLM 하나의 손실과 그래디언트를 파이썬으로 계산해 정답표로 남긴다.

    python tools/make_nplm_expected.py

결과: data/nplm_expected.csv

왜 이렇게 작은가
----------------
크기가 아니라 **수식**이 맞는지 보는 것이 목적이다. 어휘 7, 문맥 2,
임베딩 3, 은닉 4 짜리면 사람이 손으로도 따라갈 수 있다.
버그는 대개 수식에 있지 크기에 있지 않다.

가중치를 난수로 안 채우는 이유
------------------------------
C++ 의 xorshift 를 파이썬에서 똑같이 구현하면 되긴 한다. 그런데 그러면
**난수 생성기가 맞는지**까지 이 검사에 얽힌다. 검사는 한 번에 한 가지만
봐야 한다. 그래서 정수 나눗셈으로 정해지는 값을 쓴다.

    ((i * 31 + 17) % 41 - 20) / 40.0

정수 연산과 나눗셈 한 번이라 두 언어에서 **비트까지 같은 값**이 나온다.
"""

import os
import sys

import numpy as np

OUT_PATH = os.path.join("data", "nplm_expected.csv")

VOCAB = 7
CONTEXT = 2
EMBED = 3
HIDDEN = 4

CONTEXT_TOKENS = [2, 5]
TARGET = 3


def Fill(Count, Multiplier, Offset, Modulus, Half, Divisor):
    Values = np.empty(Count, dtype=np.float64)
    for i in range(Count):
        Values[i] = ((i * Multiplier + Offset) % Modulus - Half) / Divisor
    return Values


def main():
    # ---- 가중치 ----
    Embedding = Fill(VOCAB * EMBED, 31, 17, 41, 20, 40.0).reshape(VOCAB, EMBED)
    HiddenW = Fill(HIDDEN * CONTEXT * EMBED, 23, 11, 37, 18, 36.0).reshape(
        HIDDEN, CONTEXT * EMBED)
    HiddenB = Fill(HIDDEN, 13, 5, 29, 14, 28.0)
    OutputW = Fill(VOCAB * HIDDEN, 19, 7, 43, 21, 42.0).reshape(VOCAB, HIDDEN)
    OutputB = Fill(VOCAB, 29, 3, 31, 15, 30.0)

    # ---- 순전파 ----
    Embedded = np.concatenate([Embedding[Token] for Token in CONTEXT_TOKENS])

    HiddenPre = HiddenW @ Embedded + HiddenB
    HiddenOut = np.tanh(HiddenPre)

    Scores = OutputW @ HiddenOut + OutputB

    Shifted = Scores - np.max(Scores)
    Exps = np.exp(Shifted)
    Probabilities = Exps / np.sum(Exps)

    Loss = -np.log(Probabilities[TARGET])

    # ---- 역전파 ----
    GradScores = Probabilities.copy()
    GradScores[TARGET] -= 1.0

    GradOutputW = np.outer(GradScores, HiddenOut)
    GradOutputB = GradScores.copy()

    GradHiddenOut = OutputW.T @ GradScores
    GradHiddenPre = GradHiddenOut * (1.0 - HiddenOut ** 2)

    GradHiddenW = np.outer(GradHiddenPre, Embedded)
    GradHiddenB = GradHiddenPre.copy()

    GradEmbedded = HiddenW.T @ GradHiddenPre

    GradEmbedding = np.zeros_like(Embedding)
    for k, Token in enumerate(CONTEXT_TOKENS):
        GradEmbedding[Token] += GradEmbedded[k * EMBED:(k + 1) * EMBED]

    # ---- 저장 ----
    os.makedirs("data", exist_ok=True)

    Rows = []
    Rows.append((0, 0, float(Loss)))

    def Append(Kind, Array):
        Flat = np.asarray(Array).reshape(-1)
        for i, Value in enumerate(Flat):
            Rows.append((Kind, i, float(Value)))

    Append(1, GradEmbedding)
    Append(2, GradHiddenW)
    Append(3, GradHiddenB)
    Append(4, GradOutputW)
    Append(5, GradOutputB)

    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as Out:
        Out.write("Kind,Index,Value\n")
        for Kind, Index, Value in Rows:
            Out.write(f"{Kind},{Index},{Value:.17g}\n")

    print(f"완료 : {OUT_PATH}  {len(Rows)}줄")
    print()
    print(f"  어휘 {VOCAB}, 문맥 {CONTEXT}, 임베딩 {EMBED}, 은닉 {HIDDEN}")
    print(f"  문맥 토큰 {CONTEXT_TOKENS}, 정답 {TARGET}")
    print(f"  손실 = {Loss:.17g}")
    print()
    print("  확률 =", " ".join(f"{P:.6f}" for P in Probabilities))
    print("  그래디언트 합 =", f"{np.sum(GradScores):.3e}", "(0 이어야 한다)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
작은 트랜스포머 하나의 점수(로짓)를 NumPy 로 계산해 정답표로 남긴다.

    python tools/make_transformer_expected.py

결과: data/transformer_expected.csv

왜 PyTorch 가 아닌가
--------------------
원래 계획은 PyTorch 와 대조하는 것이었다. 그런데 그러면 200MB 짜리 의존성이
하나 더 붙는다. A9 에서 Kiwi 를 받은 것만으로도 저장소가 무거워졌다.

대신 C3 에서 쓴 방식을 그대로 쓴다 — **NumPy 로 참조 구현을 따로 쓴다.**
주의할 점이 하나 있다. 같은 사람이 두 번 쓰면 **같은 오해를 두 번 쓸 수**
있다. 그래서 이 파일은 C++ 코드를 보지 않고 공개된 수식에서 바로 쓴다.
그리고 C++ 쪽은 for 루프로, 이쪽은 행렬 연산으로 — 모양이 다르게 쓴다.

가중치
------
C3 과 같은 방식. 정수 나눗셈으로 정해지는 값이라 두 언어에서 비트까지 같다.
"""

import os
import sys

import numpy as np

OUT_PATH = os.path.join("data", "transformer_expected.csv")

VOCAB = 11
MODEL = 8
HEADS = 2
HIDDEN = 16
LAYERS = 2
MAX_LENGTH = 12

TOKENS = [3, 7, 1, 9, 0, 4]

EPSILON = 1e-6


def Fill(Count, Multiplier, Offset, Modulus, Half, Divisor):
    Values = np.empty(Count, dtype=np.float64)
    for i in range(Count):
        Values[i] = ((i * Multiplier + Offset) % Modulus - Half) / Divisor
    return Values


def RmsNorm(X, Gain):
    Scale = 1.0 / np.sqrt(np.mean(X * X, axis=-1, keepdims=True) + EPSILON)
    return X * Scale * Gain


def Silu(X):
    return X / (1.0 + np.exp(-X))


def Softmax(X):
    Shifted = X - np.max(X, axis=-1, keepdims=True)
    Exps = np.exp(Shifted)
    return Exps / np.sum(Exps, axis=-1, keepdims=True)


def main():
    Head = MODEL // HEADS
    Length = len(TOKENS)

    # ---- 가중치. C++ 쪽과 같은 식으로 채운다 ----
    TokenEmbedding = Fill(VOCAB * MODEL, 31, 17, 41, 20, 40.0).reshape(VOCAB, MODEL)
    PositionEmbedding = Fill(MAX_LENGTH * MODEL, 23, 11, 37, 18, 36.0).reshape(
        MAX_LENGTH, MODEL)
    HeadWeight = Fill(MODEL * VOCAB, 19, 7, 43, 21, 42.0).reshape(MODEL, VOCAB)

    Layers = []
    for L in range(LAYERS):
        Base = (L + 1) * 7
        Layers.append({
            "Q": Fill(MODEL * MODEL, 13 + Base, 5, 29, 14, 28.0).reshape(MODEL, MODEL),
            "K": Fill(MODEL * MODEL, 17 + Base, 3, 31, 15, 30.0).reshape(MODEL, MODEL),
            "V": Fill(MODEL * MODEL, 11 + Base, 9, 37, 18, 36.0).reshape(MODEL, MODEL),
            "P": Fill(MODEL * MODEL, 23 + Base, 1, 41, 20, 40.0).reshape(MODEL, MODEL),
            "Up": Fill(MODEL * HIDDEN, 29 + Base, 13, 43, 21, 42.0).reshape(MODEL, HIDDEN),
            "Down": Fill(HIDDEN * MODEL, 31 + Base, 7, 47, 23, 46.0).reshape(HIDDEN, MODEL),
        })

    AttentionGain = np.ones(MODEL)
    FeedGain = np.ones(MODEL)
    FinalGain = np.ones(MODEL)

    # ---- 순전파 ----
    X = TokenEmbedding[np.array(TOKENS)] + PositionEmbedding[:Length]

    Mask = np.triu(np.ones((Length, Length), dtype=bool), k=1)

    for L in range(LAYERS):
        W = Layers[L]

        Normed = RmsNorm(X, AttentionGain)

        Q = (Normed @ W["Q"]).reshape(Length, HEADS, Head).transpose(1, 0, 2)
        K = (Normed @ W["K"]).reshape(Length, HEADS, Head).transpose(1, 0, 2)
        V = (Normed @ W["V"]).reshape(Length, HEADS, Head).transpose(1, 0, 2)

        Scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(Head)
        Scores = np.where(Mask[None, :, :], -1e30, Scores)

        Weights = Softmax(Scores)
        Attended = Weights @ V                       # (헤드, 위치, 헤드차원)

        Merged = Attended.transpose(1, 0, 2).reshape(Length, MODEL)
        X = X + Merged @ W["P"]

        FeedNormed = RmsNorm(X, FeedGain)
        X = X + Silu(FeedNormed @ W["Up"]) @ W["Down"]

    X = RmsNorm(X, FinalGain)
    Logits = X @ HeadWeight

    # ---- 저장 ----
    os.makedirs("data", exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as Out:
        Out.write("Position,Token,Value\n")
        for t in range(Length):
            for v in range(VOCAB):
                Out.write(f"{t},{v},{float(Logits[t, v]):.17g}\n")

    print(f"완료 : {OUT_PATH}  {Length * VOCAB}줄")
    print()
    print(f"  어휘 {VOCAB}, 모델차원 {MODEL}, 헤드 {HEADS}, "
          f"앞먹임 {HIDDEN}, 층 {LAYERS}")
    print(f"  토큰 {TOKENS}")
    print()
    print("  마지막 위치의 점수")
    print("   ", " ".join(f"{float(V):+8.4f}" for V in Logits[-1]))
    print()
    print(f"  점수의 최대 절대값 = {float(np.max(np.abs(Logits))):.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

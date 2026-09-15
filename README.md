# hangul-engine

**C/C++로 한국어 LLM 엔진을 바닥부터 만든다.**

N-그램부터 트랜스포머까지 30장에 걸쳐 직접 구현하고, 마지막에는 제품에 넣을 수 있는 엔진이 남는다.

---

## 무엇을 만드는가

이 저장소의 최종 산출물은 **30개의 예제 프로그램이 아니라 하나의 엔진**이다.

`lib/` 폴더가 그 엔진이다. 지금은 UTF-8 처리기·파일 스캐너·빈도표·샘플러·해시맵 셋·N-그램·백오프·벡터·행렬·신경망 층이 들어 있고, 장을 거치며 텐서·트랜스포머가 쌓인다.

```
배우다가 물건이 된다
A~C (학습용) ──→ D~E (엔진으로 수렴)
```

로컬에서 도는 작은 한국어 모델이라 **오프라인으로 동작**한다. 게임 NPC 대화 생성 같은 곳에 쓸 수 있다.

---

## 대상 독자

C/C++를 아는 사람. 구체적으로는 아래 세 권을 마친 정도를 가정한다.

| 선행 교재 | 이 저장소에서 쓰이는 곳 |
|---|---|
| C 문법 | 포인터·동적할당·비트 연산·파일 입출력 |
| 자료구조와 알고리듬 | 해시맵·정렬·슬라이딩 윈도우 |
| C++ 문법 | 클래스·연산자 오버로딩·이동 시맨틱·STL |

각 장 머리에 **선행 절 번호**를 적어두었으므로, 막히면 해당 절로 돌아가면 된다.

수학은 따로 요구하지 않는다. 필요한 것만 그 자리에서 설명한다.

---

## 진행 상황

| 파트 | 장 | 상태 |
|---|---|---|
| **A. N-그램** (C) | A0 ~ A9 | **완료** |
| **B. 평가 도구** (C++) | B1 ~ B3 | **완료** |
| **C. 뉴럴 네트워크** (C++) | C1 ~ C6 | **완료** |
| **D. 트랜스포머** (C++) | D1 ~ D8 | D1 ~ D7 완료 |
| **E. 사후훈련과 엔진** (C++) | E1 ~ E3 | 예정 |

챕터 단위로 연재한다. 각 장의 코드 상태는 git 태그로 남긴다.

---

## 저장소 구조

```
hangul-engine/
├─ common/        교재 도구 (검증 매크로, 타입 규약) — 엔진에는 안 들어감
├─ lib/           엔진이 될 코드 — 장을 거치며 자란다
├─ ch-A00-.../    장별 실습 코드와 본문
├─ tools/         파이썬 스크립트 (데이터 준비, 시각화)
├─ data/          코퍼스와 정답표
└─ CONVENTION.md  코딩 컨벤션
```

---

## 시작하기

### 빌드

**Visual Studio / MSVC**, **C11 / C++17**, **x64** 기준이다.

프로젝트 설정 네 가지는 [A0](ch-A00-skeleton/README.md)에 정리해두었다. `/utf-8` 옵션을 빼먹으면 한글이 깨지므로 주의.

[A4](ch-A04-hashmap/README.md)부터는 **전체 프로그램 최적화**(`/GL`)와 **링크 타임 코드 생성**(`/LTCG`)을 켠 상태를 기준으로 한다. 파일 경계를 넘는 인라인이 막히면 파일을 훑는 루프에서 30%를 잃는다.

[C5](ch-C05-simd/README.md)부터는 **OpenMP 지원**(`/openmp`)도 켠다. 안 켜면 병렬 판이 한 스레드로 돌 뿐 빌드는 된다.

[B1](ch-B01-cpp/README.md)부터는 C++ 파일이 섞인다. `lib/` 와 `common/` 의 헤더에 `extern "C"` 창구를 내두었으므로, 같은 헤더를 C에서도 C++에서도 읽을 수 있다. A파트의 C 코드는 하나도 버리지 않는다.

### 데이터 준비

A2부터 한국어 위키백과 코퍼스가 필요하다. 파이썬은 **데이터 준비와 시각화에만** 쓰고, 엔진 자체는 순수 C/C++다.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install datasets numpy matplotlib

.venv/Scripts/python tools/download_corpus.py         # data/corpus.txt (약 1.3GB)
.venv/Scripts/python tools/make_expected.py           # data/expected.csv
.venv/Scripts/python tools/make_bigram_expected.py    # data/bigram_*.csv
.venv/Scripts/python tools/make_ngram_expected.py     # data/ngram_*.csv
.venv/Scripts/python tools/make_nplm_expected.py      # data/nplm_expected.csv
.venv/Scripts/python tools/make_npy_expected.py       # data/npy_*.npy
.venv/Scripts/python tools/make_transformer_expected.py  # data/transformer_expected.csv
```

A9는 형태소 분석기 Kiwi가 따로 필요하다. 버전이 고정되어 있다.

```bash
.venv/Scripts/python tools/get_kiwi.py    # third_party/kiwi (약 500MB)
```

> 전역 파이썬에 설치하면 `huggingface_hub` 버전이 올라가면서 다른 프로젝트가 깨질 수 있다. **반드시 venv를 쓸 것.**

정답표(`data/*.csv`)는 저장소에 들어 있으므로 **검증만 할 거라면 파이썬 없이도 된다.** 코퍼스는 크기 때문에 커밋하지 않는다.

### 스냅샷 고정

코퍼스는 `wikimedia/wikipedia` / `20231101.ko`로 **버전이 고정**되어 있다.

위키백과는 계속 갱신되므로 "최신"을 받으면 사람마다 결과가 달라지고 정답표와 어긋난다. `tools/download_corpus.py`의 상수를 바꾸지 말 것.

---

## 검증

모든 장은 **직접 만든 `CHECK` 매크로**로 스스로를 검증한다. 외부 테스트 프레임워크를 쓰지 않는다.

```
검사 15개 중 15개 통과
```

훈련 코드는 틀려도 손실이 *조금* 내려가며 **조용히 틀린다**. 그래서 모델보다 자를 먼저 만들었다. 자세한 이유는 [A0](ch-A00-skeleton/README.md)에 있다.

---

## 라이선스

| 대상 | 라이선스 |
|---|---|
| **코드** (`lib/`, `common/`, `ch-*/`, `tools/`) | [MIT](LICENSE) |
| **본문** (`*.md`) | [CC BY-NC 4.0](LICENSE-DOCS) |

코드는 상용 제품에 그대로 넣어도 된다. 본문은 비영리 조건으로 공유·수정할 수 있다.

코퍼스로 쓰는 한국어 위키백과는 **CC BY-SA**이며 출처는 위키미디어 재단이다.

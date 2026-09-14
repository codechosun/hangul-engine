# B1. C++로 넘어가기

> **선행** · C++ 교재 `2.1` 첫 프로그램 / `8.1` STL 컨테이너 / `8.5` 알고리듬
> **만드는 것** · `lib/*.h` 에 `extern "C"` 창구, `ch-B01-cpp/Main.cpp`
> **예제** · B1-1 ~ B1-5

---

## 0. 결론부터

A파트에서 해시맵을 세 번 만들었다. C++에는 `std::unordered_map` 한 줄이 있다. 그러면 처음부터 C++로 할 걸 그랬나?

재보면 이렇게 나온다.

| | C (직접 만든 것) | C++ (표준 라이브러리) |
|---|---|---|
| 코드 | 271줄 | **5줄** |
| 시간 | **3.21초** | 9.16초 (2.85배 느림) |
| 메모리 | **35MB** | 88MB (2.53배) |

**C++이 더 느리고 더 뚱뚱하다.** 그런데도 B파트부터 C++로 간다.

이 장은 "C++이 좋다"를 가르치는 장이 아니다. **무엇을 얻고 무엇을 잃는지 숫자로 확인하는 장**이다.

---

## 1. 헤더에 창구를 낸다

`.cpp` 파일에서 `lib/Map.h` 를 그냥 `#include` 하면 링크가 실패한다.

```
error LNK2019: unresolved external symbol "int __cdecl MapInit(struct FMap *,unsigned __int64)"
```

A9에서 본 그 문제다. C++ 컴파일러는 `MapInit` 을 **맹글링된 이름**으로 찾는데, `Map.c` 는 C 컴파일러가 번역해서 `MapInit` 그대로 내보냈다.

A9에서는 Kiwi가 우리에게 창구를 내줬다. 이번엔 **우리가 우리 자신에게** 낸다.

```c
#ifndef MAP_H
#define MAP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { ... } FMap;
int MapInit(FMap* Map, uint64_t InitialCapacity);
...

#ifdef __cplusplus
}
#endif
#endif
```

`lib/` 와 `common/` 의 헤더 15개에 전부 넣었다. 이제 같은 헤더를 C에서도 C++에서도 읽을 수 있다.

> **`extern "C"` 를 `#include` **아래**에 두는 것**이 중요하다. 위에 두면 `<stdint.h>` 같은 표준 헤더까지 `extern "C"` 안에 들어간다. 표준 헤더는 자기 안에 C++용 선언을 따로 갖고 있는 경우가 있어서, 통째로 감싸면 깨질 수 있다.

```c
FMap Probe;
CHECK(MapInit(&Probe, 16));   // .cpp 에서 부른다
MapAdd(&Probe, 42, 7);
CHECK(MapGet(&Probe, 42) == 7);
MapFree(&Probe);
```

**A파트 코드를 하나도 안 버린다.** 섞어 쓰면서 필요한 것만 C++로 옮긴다.

---

## 2. 같은 일을 두 번 — 글자 쌍 세기

A4가 한 일을 두 가지 방법으로 한다.

```c
// C — lib/Map (직접 만든 것, 271줄)
FMap Map;
MapInit(&Map, 1024);
for (size_t i = 1; i < Tokens.size(); i++)
{
    MapAdd(&Map, PackPair(Tokens[i - 1], Tokens[i]), 1);
}
...
MapFree(&Map);   // 이 줄을 잊으면 샌다
```

```cpp
// C++ — std::unordered_map (5줄)
std::unordered_map<uint64_t, uint64_t> Map;
for (size_t i = 1; i < Tokens.size(); i++)
{
    Map[PackPair(Tokens[i - 1], Tokens[i])] += 1;
}
// 해제할 줄이 없다
```

`Map[Key] += 1` 한 줄이 A4의 `MapAdd` 전부를 대신한다. **키가 없으면 0으로 만들어 넣고, 있으면 찾아서 더한다.**

```
[B1-2] 글자 쌍 세기 — lib/Map 대 std::unordered_map

                        C — lib/Map                  C++ — unordered_map
  서로 다른 쌍          1227989                       1227989
  '의'->공백 횟수       7519110                       7519110
  시간                  3.21 초                       9.16 초  (2.85배)
  메모리                35 MB                         88 MB  (2.53배)
  코드                  271 줄                        5 줄
```

**답은 정확히 같다.** A4에서 잰 1,227,989와도 같다.

```c
CHECK(CDistinct == CppDistinct);
CHECK(CCheck == CppCheck);
```

### 왜 느린가

우연이 아니다. **표준이 그렇게 하라고 정해놨다.**

`std::unordered_map` 은 **체이닝**(chaining)으로 만들게 되어 있다. 칸마다 연결 리스트를 두고, 원소 하나마다 **노드를 따로 할당**한다.

```
버킷:  [ ] [*] [ ] [*] [*] ...
            |       |   |
           노드    노드 노드      <- 각각 따로 malloc
```

우리 `lib/Map` 은 **열린 주소**(open addressing)다. 큰 배열 하나에 값을 직접 담는다.

| | `lib/Map` | `unordered_map` |
|---|---|---|
| 원소 하나당 할당 | 없음 (배열 한 덩어리) | **노드 하나** |
| 메모리 접근 | 배열을 순차로 | **포인터를 따라간다** |
| 원소 하나 크기 | 17바이트 | 노드 = 키 8 + 값 8 + `next` 8 + 할당자 헤더 |

123만 번 `new` 를 부르고, 찾을 때마다 포인터를 따라간다. 캐시가 계속 어긋난다. **2.85배는 그 값이다.**

표준이 왜 굳이 이렇게 정했는가. **참조 안정성**(reference stability) 때문이다.

```cpp
auto& Value = Map[Key];
Map[OtherKey] = 1;        // 표가 자라도
Value += 1;               // 이 참조는 여전히 유효하다
```

열린 주소 방식에서는 표가 자라면 원소가 전부 옮겨가므로 이게 불가능하다. A6에서 "아레나가 자라면 포인터가 끊어진다"고 한 그 문제다. 표준은 **안정성을 택하고 속도를 포기했다.**

> 속도가 필요하면 열린 주소로 만든 대체품을 쓴다. `absl::flat_hash_map`(구글), `robin_hood::unordered_map`, `ankerl::unordered_dense` 같은 것들이 있고, 전부 우리가 A4에서 만든 것과 같은 구조다. **표준이 항상 최선은 아니다** — 이걸 아는 것과 모르는 것의 차이가 크다.

---

## 3. 문자열 사전

A9의 `lib/Vocab` 도 마찬가지다.

```cpp
std::unordered_map<std::string, uint32_t> Ids;
std::vector<std::string> Texts;

for (const std::string& Word : Words)
{
    auto Found = Ids.find(Word);
    if (Found == Ids.end())
    {
        Ids.emplace(Word, (uint32_t)Texts.size());
        Texts.push_back(Word);
    }
}
```

**9줄.** `lib/Vocab` 은 287줄이었다. 아레나도, `realloc` 도, 끊어진 포인터 주의사항도 전부 사라진다.

```
[B1-3] 문자열 사전 — lib/Vocab 대 unordered_map<string, uint32_t>

  서로 다른 낱말        69799                         69799
  시간                  0.008 초                      0.013 초  (1.66배)
  코드                  287 줄                        9 줄
```

여기서도 C++이 1.66배 느리다. 그런데 **0.008초와 0.013초**다. 5밀리초 차이로 287줄을 살 수 있으면 사는 게 맞다.

**속도 차이를 언제 신경 쓸 것인가**가 판단이다. 6억 번 도는 루프(B1-2)는 3배가 6초가 되지만, 20만 번 도는 루프(B1-3)는 3배여도 5밀리초다.

---

## 4. 전역 변수가 사라진다

A6에서 `qsort` 때문에 전역을 써야 했던 것을 기억할 것이다.

```c
static int GSortOrder = 0;   // 보기 좋지 않다. 멀티스레드에서는 위험하다

int CompareGramRow(const void* A, const void* B)
{
    for (int i = 0; i < GSortOrder; i++) { ... }
}
```

C++의 `std::sort` 는 **함수 객체**를 받는다. 람다는 주변 변수를 잡아간다.

```cpp
const int Mode = 0;                     // 지역 변수다

std::sort(Rows.begin(), Rows.end(),
          [Mode](const FRow& L, const FRow& R)
          {
              if (L.A != R.A) return L.A < R.A;
              if (Mode == 0) return L.B < R.B;
              return false;
          });
```

`[Mode]` 가 캡처다. 컴파일러가 `Mode` 를 멤버로 가진 이름 없는 구조체를 만들고, `operator()` 에 비교 코드를 넣는다. 전역이 필요 없고, 스레드마다 다른 값을 써도 안전하다.

그리고 **더 빠르다.**

```
[B1-4] 정렬 — qsort + 전역 대 std::sort + 람다

  시간                  1.34 초                       0.90 초  (0.67배)
  설정을 넘기는 법      파일 범위 전역                람다 캡처
```

**1.5배 빠르다.** 이유는 인라인이다.

- `qsort` 는 비교 함수를 **함수 포인터**로 받는다. 원소를 비교할 때마다 간접 호출이 일어나고, 컴파일러는 그 안을 못 본다
- `std::sort` 는 비교자를 **타입**으로 받는다(템플릿 인자). 컴파일러가 비교 코드를 정렬 루프 안에 **펼쳐 넣는다**

A7에서 `/GL` 로 되찾았던 그 비용이다. 다만 함수 포인터는 `/GL` 로도 못 되찾는다 — 어떤 함수가 올지 컴파일 시점에 모르기 때문이다. 템플릿은 **컴파일 시점에 안다.**

이게 A9에서 예고한 이야기의 결말이다. 해시맵 셋을 `void*` 와 함수 포인터로 합치면 이 비용을 낸다. 템플릿은 안 낸다.

두 정렬 결과가 같은지도 확인한다.

```c
CHECK(SameOrder == 1);
```

---

## 5. 해제하는 줄이 사라진다

A파트의 C 코드를 세어보면,

```
[B1-5] 지금까지 쓴 C 코드에서 해제가 차지하는 자리

  lib/ 의 free 호출     = 45 번
  lib/ 의 ...Free 함수  = 7 개
  이 파일(.cpp)의 free  = 0 번
```

`free` 가 45번 나온다. 그중 상당수는 **오류 처리 사다리**다.

```c
// lib/Ngram.c 에서
if (Grams == NULL || Cumulative == NULL)
{
    free(Grams);
    free(Cumulative);
    free(Rows);
    return 0;
}
...
if (ContextStart == NULL)
{
    free(Grams);
    free(Cumulative);
    free(Rows);
    return 0;
}
```

같은 세 줄이 반복된다. 그리고 **하나만 빠뜨려도 샌다.** 새 할당을 추가하면 모든 사다리를 고쳐야 한다.

C++에서는 이 코드가 **전부 사라진다.**

```cpp
{
    std::unordered_map<uint64_t, uint64_t> Map;
    std::vector<uint32_t> Tokens;
    ...
}   // 여기서 소멸자가 부린다
```

블록을 어떻게 벗어나든 — 정상 종료든, 중간 `return` 이든, 예외든 — **소멸자가 반드시 불린다.** 이것이 **RAII**(Resource Acquisition Is Initialization)다. 자원의 수명을 객체의 수명에 묶는다.

이름이 어렵지만 규칙은 한 줄이다. **생성자에서 잡고 소멸자에서 놓는다.**

> 우리가 A파트에서 `XxxInit` / `XxxFree` 짝을 맞춰 쓴 것이 RAII를 손으로 흉내낸 것이다. 다만 손으로 하면 짝을 빠뜨릴 수 있고, C++은 컴파일러가 짝을 맞춰준다.

---

## 정리

바뀐 것.

| | 전 | 후 |
|---|---|---|
| 헤더 | C 전용 | `extern "C"` 창구를 낸 C/C++ 겸용 |
| 첫 `.cpp` | 없음 | `ch-B01-cpp/Main.cpp` |
| 빌드 | `/std:c11` | `/std:c++17 /EHsc` |

잰 것.

| 항목 | C | C++ | 판정 |
|---|---|---|---|
| 해시맵 코드 | 271줄 | 5줄 | **C++ 압승** |
| 해시맵 속도 | 3.21초 | 9.16초 | C 승 (2.85배) |
| 해시맵 메모리 | 35MB | 88MB | C 승 (2.53배) |
| 정렬 속도 | 1.34초 | 0.90초 | **C++ 승 (1.5배)** |
| 해제 코드 | 45곳 | 0곳 | **C++ 압승** |

알아둘 것.

- **`extern "C"` 는 `#include` 아래에 둔다.** 표준 헤더를 감싸면 깨질 수 있다
- **`unordered_map` 은 체이닝이 강제되어 있다.** 참조 안정성을 위해 속도를 포기한 설계
- **`std::sort` 가 `qsort` 보다 빠르다.** 비교자가 타입이라 인라인된다
- **RAII 는 "생성자에서 잡고 소멸자에서 놓는다"** 한 줄이다
- **속도 차이를 언제 신경 쓸지 판단한다.** 6억 번 도는 루프와 20만 번 도는 루프는 다르다

쓴 문법.

- `std::vector`, `std::unordered_map`, `std::string` — C++ 교재 `8.1`
- `std::sort` 와 람다 — C++ 교재 `8.5`
- 범위 기반 for (`for (const auto& X : Container)`)
- `std::chrono` 로 시간 재기

---

## 연습문제

**1.** `std::unordered_map` 에 `reserve` 를 미리 부르면 얼마나 빨라지는가?

<details>
<summary>해답</summary>

```cpp
std::unordered_map<uint64_t, uint64_t> Map;
Map.reserve(1300000);          // 쌍이 123만 개쯤 나온다는 걸 안다
```

빨라진다. **리해싱이 사라지기 때문**이다. 기본 상태로 두면 123만 개가 들어가는 동안 버킷 배열이 여러 번 다시 잡히고, 그때마다 모든 원소의 해시를 다시 계산해 옮긴다.

그런데 **격차가 다 메워지지는 않는다.** 노드를 하나씩 할당하는 비용과 포인터를 따라가는 비용은 `reserve` 로 못 줄인다. 그게 이 컨테이너의 구조 자체이기 때문이다.

`std::vector::reserve` 는 얘기가 다르다. 우리 코드에도 있다.

```cpp
std::vector<uint32_t> Tokens;
Tokens.reserve(700 * 1000 * 1000);
```

이걸 빼면 6억 개를 넣는 동안 `vector` 가 **약 30번** 두 배로 자라고, 그때마다 전체를 복사한다. 옮기는 총량은 최종 크기의 두 배쯤이라 점근적으로는 같지만(A4의 분할 상환), 2.4GB를 두 번 복사하는 것은 실제로 느리다.

**크기를 미리 알면 알려준다.** 이건 C에서도 C++에서도 같다.

</details>

**2.** `Map[Key] += 1` 은 키가 없을 때 무엇을 하는가? `find` 로 쓰면 무엇이 달라지는가?

<details>
<summary>해답</summary>

`operator[]` 는 **없으면 만든다.** 값을 `uint64_t()` 즉 `0` 으로 초기화해 넣고 그 참조를 돌려준다. 그래서 `+= 1` 이 그냥 동작한다.

편한 만큼 함정도 있다.

```cpp
if (Map[Key] > 0) { ... }     // 없던 키가 0 으로 **생겼다**
```

읽기만 하려던 코드가 표를 키운다. 반복문 안에서 이러면 표가 조용히 부풀어 오른다. 읽기만 할 때는 `find` 를 쓴다.

```cpp
auto Found = Map.find(Key);
if (Found != Map.end() && Found->second > 0) { ... }
```

`at(Key)` 도 있다. 없으면 `std::out_of_range` 예외를 던진다. 우리는 예외를 안 쓰기로 했으므로(결정 15) `find` 쪽이다.

속도도 다르다. 세는 용도라면 `operator[]` 가 **한 번만 찾는다.** `find` 로 확인하고 없으면 `emplace` 하면 **두 번 찾는다.** B1-3의 문자열 사전이 그렇게 돼 있는데, 거기서는 번호를 `Texts.size()` 로 정해야 해서 어쩔 수 없다.

C++17에는 `try_emplace` 가 있어서 한 번에 끝난다.

```cpp
auto [It, Inserted] = Ids.try_emplace(Word, (uint32_t)Texts.size());
if (Inserted) { Texts.push_back(Word); }
```

</details>

**3.** `lib/Map` 이 2.85배 빠른데도 B파트부터 C++로 가는 것이 맞는가?

<details>
<summary>해답</summary>

**성능만 보면 아니다.** 그리고 그게 이 표를 실은 이유다.

세 가지를 저울에 올려야 한다.

**첫째, 그 3배가 어디서 나는가.** B1-2의 6억 번 루프는 실제로 6초가 붙는다. 하지만 C파트부터 시간의 대부분은 **행렬 곱셈**이 먹는다. 거기서 `unordered_map` 은 아예 안 쓰인다. **병목이 아닌 곳의 3배는 0이다.**

**둘째, 앞으로 만들 것이 무엇인가.** D파트의 `Tensor` 는 차원 정보를 들고, 복사하면 비싸고, 함수에서 돌려줘야 하고, 연산자를 오버로딩해야 한다. C로 만들면 `TensorInit`/`TensorFree`/`TensorCopy`/`TensorMove` 를 손으로 짝 맞춰 쓰고, **그 짝을 빠뜨리는 순간 조용히 샌다.** C++은 이걸 컴파일러가 강제한다.

**셋째, 성능이 정말 필요한 곳은 어차피 손으로 쓴다.** C5에서 GEMM을 SIMD로 짜고, D5에서 비트를 직접 패킹한다. 그 코드는 C++로 써도 C와 똑같이 생겼다. **C++은 위쪽(설계)에서 값을 하고, 아래쪽(커널)은 어느 언어든 같다.**

그리고 필요하면 **A파트 코드를 그대로 쓰면 된다.** `extern "C"` 창구를 내둔 이유가 이것이다. 실제로 `lib/Map` 은 B파트 이후에도 살아 있다.

정리하면 이렇다.

> C++로 가는 이유는 빠르기 때문이 아니라, **틀리기 어렵고 고치기 쉽기 때문**이다. 빠른 게 필요한 자리는 그때 손으로 쓴다.

</details>

---

## 다음 장

**B2. 퍼플렉서티와 수치 안정성.**

A8에서 벽에 부딪혔다. 할인율을 0.75로 정했는데, **0.5가 더 나은지 0.9가 더 나은지 말해주는 숫자가 없었다.** 베끼기는 줄었는데 문장이 짧아졌고, 둘 중 무엇이 더 중요한지 판단할 근거가 없었다.

B2에서 그 자를 만든다. **퍼플렉서티**(perplexity)는 "모델이 다음 글자를 고를 때 평균 몇 개 중에서 헷갈리는가"를 하나의 숫자로 말해준다. 완벽한 모델이면 1, 27,663개 중에서 아무거나 찍는 모델이면 27,663.

만드는 과정에서 부동소수점의 바닥을 본다.

- 확률을 1,000개 곱하면 `double` 에서도 **0이 된다** (언더플로)
- 그래서 로그를 쓴다. 그런데 확률이 0이면 로그가 **음의 무한대**다
- 무한대가 한 번 섞이면 평균이 통째로 무한대가 된다

**log-sum-exp** 라는 기법으로 이 셋을 한꺼번에 푼다. C파트의 softmax에서 똑같은 문제가 다시 나오므로, 여기서 제대로 만들어두면 두 번 만들지 않아도 된다.

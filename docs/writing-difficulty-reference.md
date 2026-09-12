# Writing 난이도 생성 기준 구현

## 범위와 버전

이 문서는 Daily Writing의 TRANSLATION / GUIDED / FREE **문제 생성**에 적용한다.
학습자 답변 평가 점수, BE의 Profile 계산, 다른 서비스의 생성 정책은 변경하지 않는다.

- 실행 정책: `writing-difficulty-reference-v4-single-semantic`
- 생성 프롬프트: `writing-generation-difficulty-reference-v2`
- 구조 규격: `writing-difficulty-spec-v1`
- 기존 요청/응답 DTO와 API 경로는 유지한다.

## 실행 흐름

```text
기존 resolver로 targetBand / difficulty / order 계획
    → WritingDifficultySpec
    → Generator (기존 모델 등급)
    → 실제 후보의 구조 / 문자 / 중복 검사
    → Mini 의미 검증 1회 (본문·가이드·설명·실제 난이도)
    → 코드의 승인 결정
        ├─ PASS + 목표 band 일치: 승인
        ├─ 명확한 품질 문제 / 다른 유형 / 확정적인 다른 band: 폐기
        ├─ 명시적인 UNSURE 또는 목표를 포함한 인접 band 경계:
        │     이전 답·목표 없이 독립 Mini 판정 최대 1회
        └─ 설명의 언어만 오류: 기존 설명 전용 보정 + 새 설명 Mini 검증
    → 승인된 내용 hash 재확인 / 서버 제어값 기록 / 최종 DTO 재검증
```

정상 경로는 생성 1회 + Mini 1회다. 이는 최소 정상 경로의 **논리 호출 수**이며,
생성 재시도·검증 통신 복구·명시적 불확실·설명 복구가 필요한 경로의 상한을 뜻하지 않는다.
Provider 내부의 API 재시도·fallback도 별도일 수 있다.

## 코드가 확인하는 것과 확인하지 않는 것

`difficulty_spec.py`의 hard constraints는 실제 문자열/배열에서 계산한다.

- NFC 정규화 기준 문자 수(내부 공백 포함), 구두점/개행으로 나눈 비어 있지 않은 표면 단위 수.
- 각 가이드 배열의 개수, 개별 항목과 전체 가이드의 길이, 설명 길이.
- 기존 모드별 필수/금지 필드, 선택 키워드 ID 집합, 알려진 문자 조건, 중복 검사를 유지한다.
- 최종 응답에서도 항목 수·순서·난이도 분배·유형·구조 규격을 다시 확인한다.

**표면 단위 수는 절·문장 의미·intent 개수와 같지 않다.** 약어 등은 더 많이 나뉠 수 있다.
문자 존재 검사는 실제 언어 식별기가 아니다. 키워드 ID가 유효하다고 내용의 관련성이
증명되는 것도 아니다. 구문 관계·자연스러움·의미 단위·register·정답 유출·배경지식 필요성은
Mini가 실제 내용을 보고 판단한다. Generator가 만든 tag를 실제 조건 충족의 증거로 삼지 않는다.

현재 길이/개수 한계와 1~5 band 서술은 버전 관리되는 초기 편집 정책이다.
학습자 응답으로 보정된 난이도나 외부 시험 등급을 뜻하지 않는다.
길이에 따라 어려운 band를 자동 부여하거나 쉬운 문장을 길게 늘려 승격하지 않는다.
GUIDED/FREE의 안내문 길이를 요구되는 산출 난이도로 간주하지 않는다.

## 난이도 검증과 승인

Generator는 선택된 난이도 규격을 받는다. Reviewer는 다섯 band 전체의 공통 설명을 받고,
선택된 targetBand·Profile·Generator 난이도 메타데이터·이전 판정은 받지 않는다.
Reviewer의 확정 band와 서버 targetBand가 일치해야 승인한다.

- `ASSESSED`: 하나의 band를 확정. 목표와 다르면 폐기하고 재투표하지 않는다.
- `BORDERLINE`: 인접한 두 band. 목표가 그 둘에 포함된 경우만 최종 독립 판정을 한 번 요청한다.
- `UNSURE`: 확정 band가 없다. 마지막 독립 판정도 불확실하면 승인하지 않는다.
- 명확한 hard violation은 낮거나 높은 confidence와 무관하게 폐기한다.

`confidence`, `difficultyConfidence`는 **진단 전용**이다. `0.0`, `0.72`, `0.73` 같은 값은
승인·거부·재시도 분기를 바꾸지 않는다. 필드는 필수이며 숫자 또는 `null`을 반환한다.
문자열·bool·NaN·Infinity·범위 밖 숫자는 여전히 계약 오류로 처리한다.
유효한 의미 판정의 낮은 자기신뢰도를 timeout으로 재분류하지 않는다.

## 의미 검증 계약

Mini는 다음 다섯 criterion을 각 한 번씩 평가한다.

1. `ORIGIN_LANGUAGE`
2. `TASK_VALIDITY`
3. `ANSWER_LEAK`
4. `NATURALNESS`
5. `NOTE_QUALITY`

각 항목은 `PASS / FAIL / UNSURE`, 해당 segment ID, 구체적인 실패 issue를 사용한다.
전체 verdict와 항목 결과·issues·난이도 상태가 모순되면 정상 의미 판정으로 취급하지 않는다.

서버가 실제 필드에 `O1`(본문), `F1...`(facts), `I1...`(intents), `C1...`(constraints),
`N1`(설명)을 부여한다. 자유문자열 인용 대신 허용된 ID를 반환받고 집합·참조 범위를 검사한다.
**참조 ID가 존재한다는 것은 출처 연결만 보장하며 의미 판단의 정확성을 보장하지 않는다.**

설명도 사용자에게 보이므로 정상 Mini 호출에 함께 들어간다. 난이도 근거로 N1을 사용할 수
없고 설명에 내부 band 주장이나 검증 지시가 있으면 거부한다. 그러나 같은 모델 호출에서
설명을 보았다는 영향 자체를 코드로 없앨 수는 없다. 이것을 완전한 심리적 독립성으로 주장하지 않는다.

## 설명 언어 복구

본문이 승인되고 설명 언어만 문제일 때에만 Nano가 설명 하나를 보정한다(최대 한 번).
보정 응답에는 본문·가이드·band·순서를 바꿀 수 있는 필드를 허용하지 않는다.
새 설명은 새 hash·검증 ID로 Mini에게 별도 검증받는다. 이전 설명의 승인 결과를 재사용하지 않는다.
보정 실패나 새 설명의 `UNSURE`/정답 유출/내용 불일치가 있으면 승인하지 않는다.
정상 설명에는 별도 note verifier를 호출하지 않는다.

## 장애·시간 예산·취소

- 기본 생성 후보는 슬롯당 라운드 최대 4회, 라운드당 최대 2개다.
- 전체 요청 240초, 개별 생성/검증 30초의 기존 기본값을 유지한다.
- 주 검증의 통신·계약 오류는 **같은 후보/같은 단계**에서 최대 2회 시도한다.
- 최종 adjudicator는 후보당 최대 1회 호출한다. 장애여도 adjudicator를 반복하지 않는다.
- 정상적인 `UNSURE`는 주 검증을 재호출하지 않고 승인 정책이 최종 판정으로 보낸다.
- 필수 검증 장애가 지속되면 생성부터 다시 반복하지 않고 요청을 종료한다.
- 503 검증 불가 / 502 설정 오류 / 504 전체 제한 / 422 후보 기준 미충족 구분을 유지한다.
- 외부 요청 취소/전체 timeout은 await 중인 작업으로 전달되고 미검증 결과를 반환하지 않는다.
- 후보와 판정 재사용은 현재 AI 요청 범위다. 서버 재시작을 넘는 후보 저장 기능은 없다.

검증 단계를 줄였어도 실제 부적합 후보나 외부 장애가 반복되면 오래 걸리거나 실패할 수 있다.
특정 실제 생성시간·성공률·정확도를 보장하지 않는다.

## 관측 로그

기존 JSON diagnostics에 `writing.spec.checked`, `writing.acceptance.decided`,
`writing.adjudication.requested`, `difficulty_spec_version`, `confidence_used_for_acceptance=false`를 포함한다.
스키마 오류의 안전한 필드 경로, stage별 호출·시간, provider/meta가 있으면 토큰 수를 유지한다.
원문·프롬프트·Profile·키·원시 예외 문자열·검증 입력값은 로깅하지 않는다.
`outcome=VALID`는 응답 계약이 유효하다는 뜻이다. 실제 승인 여부는 acceptance/최종 summary를 확인한다.

## Progressive 생성 및 다음 단계

BE가 한 문항을 요청하면 AI도 승인된 한 문항을 반환한다. 전체 다섯 문항을 기다리도록 바꾸지 않는다.
BE 저장/polling은 이번 패치 범위 밖이며, 실제 연결 환경에서 첫 문항 노출·다음 문항 생성·부분 실패
재시도를 확인해야 한다.

먼저 이 Writing 기준 구현을 로컬 전체 검사와 실제 모델 smoke로 검증한다. 그 뒤 공통 계약을
추출하고 Listening·Reading·Vocabulary·Speaking·Level Test 문제풀 생성에 통합 적용한다.
이번 패치가 그 후속 서비스의 변경까지 완료했다는 뜻은 아니다.

# AIproject : 비판적 재평가 텍스트를 통한 planner Model 성능 개선 듀얼 시스템 VLA 모델 개발

## 연구 개요

본 프로젝트는 OpenVLA가 생성한 행동을 그대로 수행하지 않고, Critic Actor가 행동을 분석하여 자연어 형태의 비판적 설명(Critique)을 생성한 뒤 수정된 Action Token을 다시 생성하도록 설계된 Dual-System Vision-Language-Action(VLA) 모델이다.

기존 Planner 기반 로봇 제어 모델은 행동 생성 이후 행동의 적절성을 재평가하는 과정이 부족하다. 본 연구는 Planner가 생성한 행동을 Critic Actor가 분석하고 수정하는 구조를 제안하여 행동 수행 이전에 잠재적인 문제를 탐지하고 개선할 수 있도록 하였다.

실제 로봇 환경 구축의 비용과 제약을 고려하여 LIBERO 시뮬레이션 환경에서 실험을 수행하였으며, GRPO(Group Relative Policy Optimization)를 이용해 행동 수정 정책(Action Refinement Policy)을 학습하였다.

---

# 시스템 구조

<p align="center">
  <img src="images/system_architecture.png" width="900">
</p>

본 연구는 Planner와 Critic Actor로 구성된 Dual-System 구조를 사용한다.

Planner는 현재 환경 이미지와 작업 명령(Task Instruction)을 입력받아 초기 행동(Action Token)을 생성한다.

Critic Actor는 Planner의 행동을 입력으로 받아 행동의 문제점을 자연어 형태의 Critique로 설명하고, 이를 기반으로 수정된 Action Token을 생성한다.

수정된 행동은 LIBERO 환경에서 평가되며, Reward Function을 통해 계산된 보상을 이용하여 GRPO 강화학습을 수행한다.

---

# 핵심 구조

<p align="center">
  <img src="images/model_structure.png" width="900">
</p>

Planner가 생성한 Action Token은 Projection Layer를 통해 Critic Actor의 임베딩 공간으로 변환된다.

Critic Actor는 다음 세 가지 정보를 동시에 입력받는다.

- 현재 환경 이미지
- 작업 명령(Task Instruction)
- Planner가 생성한 Action Token

이를 기반으로

- 비판적 텍스트(Critique)
- 수정된 Action Token

을 생성한다.

---

# 연구 기여점

본 연구의 주요 기여는 다음과 같다.

1. Planner와 Critic Actor로 구성된 Dual-System VLA 구조를 제안하였다.

2. 자연어 기반 Critique를 활용한 행동 수정(Action Refinement) 구조를 구현하였다.

3. OpenVLA Action Token을 Critic Actor에 이식하기 위한 Tokenizer 확장 및 Projection Layer를 구현하였다.

4. Critique 생성과 Action Refinement를 동시에 수행하는 구조를 설계하였다.

5. LoRA와 4bit Quantization을 적용하여 제한된 GPU 환경에서도 학습이 가능하도록 구현하였다.

---

# Action Token 이식 과정

OpenVLA와 SmolVLM은 서로 다른 토크나이저와 임베딩 공간을 사용한다.

따라서 Planner가 생성한 Action Token을 Critic Actor가 이해할 수 있도록 다음 과정을 수행하였다.

1. OpenVLA의 Action Embedding 추출

2. Projection Layer를 이용하여 4096차원(OpenVLA)에서 960차원(SmolVLM)으로 변환

3. SmolVLM 토크나이저에 256개의 Action Token 추가

4. 이미지 임베딩, 텍스트 임베딩, Action 임베딩을 결합하여 입력

이를 통해 Critic Actor는

- 현재 이미지
- 작업 명령
- Planner의 행동

을 동시에 고려하여 비판적 텍스트와 수정된 행동을 생성할 수 있다.

# Research Motivation

기존 Planner 기반 로봇 제어 모델은 행동 생성 이후 재검토 과정이 존재하지 않는다.

예시:

* 로봇 팔이 목표 물체보다 지나치게 높은 위치로 이동
* 물체를 잡기 전에 그리퍼를 닫음
* 충돌 위험이 높은 경로 선택
* 목표 객체를 놓친 상태에서도 행동 수행

Planner는 이러한 문제를 포함한 행동을 그대로 실행할 수 있다.

본 연구에서는 행동 수행 이전에 Critique 과정을 추가하여 행동을 평가하고 수정하는 구조를 설계하였다.

---

# Research Goal

### 1. Planner Action Analysis

OpenVLA가 생성한 Action Token을 입력으로 받아 행동을 분석한다.

### 2. Natural Language Critique Generation

행동의 문제점과 개선 방향을 자연어 형태로 생성한다.

Example

```text
[CRITIQUE]
The robot arm is too high and may miss the target object.
[/CRITIQUE]
```

### 3. Action Refinement

Critique를 기반으로 수정된 Action Token을 생성한다.

### 4. Reinforcement Learning

LIBERO 환경에서 얻은 보상을 이용하여 행동 수정 정책을 지속적으로 개선한다.

---

# Experimental Environment

## OpenVLA

Planner 역할 수행

입력:

* RGB Image
* Task Instruction

출력:

* Planner Action Tokens (7 Tokens)

모델:

* OpenVLA-7B Finetuned on LIBERO-10

---

## Qwen2.5-VL

초기 Critic Actor 모델

특징:

* Vision-Language 입력 지원
* Critique 생성
* Action Refinement 수행

단점:

* 높은 VRAM 사용량
* 강화학습 시 OOM 발생 가능

---

## SmolVLM2

최종 Critic Actor 모델

특징:

* Critique 생성
* Action Refinement 수행
* Vision-Language 입력 지원
* 저용량 모델

모델:

* SmolVLM2-500M-Video-Instruct

선정 이유:

* Qwen 대비 낮은 VRAM 사용량
* LoRA 적용 용이
* 강화학습 실험 가능

---

## LoRA

Low-Rank Adaptation

전체 모델을 업데이트하지 않고 일부 행렬만 학습한다.

장점:

* VRAM 절약
* 빠른 학습
* 적은 GPU 요구사항

---

## GRPO

Group Relative Policy Optimization

동일 상태에서 여러 행동 후보를 생성하고 상대적 보상을 통해 정책을 업데이트하는 강화학습 알고리즘이다.

본 연구에서는 Critique 기반 Action Refinement 정책 학습에 사용하였다.

---

## LIBERO

Robot Manipulation Benchmark

실제 로봇 없이 다양한 조작 작업을 평가할 수 있는 시뮬레이션 환경이다.

사용 환경:

* LIBERO-10

---

# Reward Function

보상 함수는 다음 요소를 결합하여 계산한다.

| Component        | Reward           |
| ---------------- | ---------------- |
| Task Success     | +1.0             |
| Collision        | -0.5             |
| Time Penalty     | -0.01 × overtime |
| Stability Reward | +0.3             |
| Invalid Action   | -1.0             |

최종 보상:

```text
Reward ∈ [-1.0, 2.0]
```

---

# Project Structure

```text
AIproject
│
├── openvla_planner
│   ├── openvla_inference_code.py
│   └── action_tokenizer.py
│
├── qwen_actor
│   ├── actor_action_tokenizer.py
│   ├── projection_layer.py
│   └── actor_model.py
│
├── SmolVLM_actor
│   ├── smol_action_tokenizer.py
│   ├── smol_projection_layer.py
│   └── smol_actor_model.py
│
├── train
│   ├── train.py
│   ├── smol_train.py
│   └── smol_sft.py
│
├── SFT
│   └── SFT.py
│
├── assets
│   └── make_embeddings.py
│
├── checkpoints
│   ├── sft
│   └── sft2
│
└── logs
```

---

# 📁 openvla_planner

### openvla_inference_code.py

OpenVLA Planner 추론 서버.

* ZeroMQ 기반 서버 실행
* Planner 전용 추론 수행
* CPU 환경에서 동작
* OpenVLA Action Token 생성

Model

* OpenVLA-7B Finetuned on LIBERO-10

Input

* RGB Image
* Task Instruction

Output

* Planner Action Tokens (7 Tokens)

---

### action_tokenizer.py

OpenVLA 원본 Action Tokenizer.

본 프로젝트에서 사용하는 Action Token 구조를 확인하기 위해 유지하였다.

주요 역할

* Action Token 정의 확인
* Action Embedding 추출
* Token ↔ Action 변환 확인

---

# 📁 qwen_actor

Qwen 기반 Critic Actor 구현.

### actor_action_tokenizer.py

* OpenVLA Action Token 256개 추가
* Planner Action Embedding 로드
* Projection Layer 적용
* Qwen 입력 임베딩과 결합

### projection_layer.py

```text
OpenVLA Hidden Size : 4096
Qwen Hidden Size    : 2048
```

LLaVA Projection Layer 구조를 참고하였다.

### actor_model.py

* Qwen2.5-VL 기반
* 4bit Quantization
* LoRA Fine-Tuning
* ZeroMQ 통신
* Critique 생성
* Action Token 수정

---

# 📁 SmolVLM_actor

Qwen 모델의 높은 VRAM 사용량 문제를 해결하기 위해 구현하였다.

### smol_action_tokenizer.py

* Action Token 등록
* Action Embedding 초기화
* Action Token 변환

### smol_projection_layer.py

```text
OpenVLA Hidden Size : 4096
SmolVLM Hidden Size : 960
```

LLaVA Projection Layer를 참고하였다.

### smol_actor_model.py

* SmolVLM2-500M
* 4bit Quantization
* LoRA
* ZeroMQ
* Critique 생성
* Action Token 수정

---

# 📁 train

강화학습 실행 파일

### train.py

* LIBERO 환경 생성
* Rollout 수집
* Reward 계산
* GRPO 학습
* Actor 업데이트

### smol_train.py

RLinf 구조 참고

주요 함수

```python
collect_rollout()
compute_grpo_loss()
```

### smol_sft.py

GRPO 이전 SFT 단계

목적

* Critique 형식 학습
* Action Token 형식 학습
* RL 안정성 향상

---

# 📁 assets

### make_embeddings.py

OpenVLA Action Embedding 추출

생성 파일

```text
openvla_action_embeddings.pt
```

---

# 📁 checkpoints

### sft

초기 체크포인트

* 비전 인코더 미사용
* 현재 사용하지 않음

### sft2

최종 체크포인트

* 비전 인코더 포함
* Critique 생성 가능
* Action Token 생성 가능

---

# 📁 logs

저장 항목

* Reward
* Success Rate
* Loss
* Learning Rate

---

# Expected Contributions

* Critique 기반 행동 수정 구조 제안
* Planner-Critic Dual System 제안
* 자연어 기반 행동 재평가 가능성 검증
* 실제 로봇 없이 LIBERO 환경 검증
* 실제 로봇 환경으로 확장 가능성 제시

---

# Limitations

* 실제 로봇 환경 검증 미수행
* Critique 품질에 따른 성능 편차 존재
* 장기 작업(Long Horizon Task)에 대한 추가 검증 필요

---

# Future Work

* Real Robot Evaluation
* Multi-Step Critique Generation
* Human Feedback Integration
* Larger Critic Models
* Hierarchical Planning

```
```

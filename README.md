# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

이곳은 GPT 모델을 훈련/미세 조정하기 위한 가장 간단하고 빠른 레포지토리입니다!

(한국어로 현재 번역 작업을 진행하고 있습니다. 코드 중 한국어로 번역 및 기본적인 GPT-2의 구조를 학습하고 싶으신 분은 `gpt_dev.ipynb`를 참고하세요. )

 [minGPT](https://github.com/karpathy/minGPT)의 리라이트 버전으로, 훈련보다는  성능을 우선시합니다. 여전히 개발 중이지만, 현재 `train.py` 파일은 GPT-2 (124M)를 OpenWebText에서 단일 8XA100 40GB 노드에서 약 4일 동안 훈련하여 재현합니다. 코드 자체는 간결하고 읽기 쉬워서, `train.py`는 약 300줄의 Boiler Plate 훈련 루프이며, 
 `model.py`는 약 300줄의 GPT 모델을 구현하였습니다. 
  OpenAI에서 GPT-2 사전 훈련된 가중치를 로드할 수도 있습니다. 

![repro124m](assets/gpt2_124M_loss.png)

코드가 매우 간단하기 때문에 필요에 맞게 수정하거나, 새로운 모델을 처음부터 훈련하거나, 사전 훈련된 체크포인트를 미세 조정하는 것이 매우 쉽습니다 (예: 현재 사용 가능한 가장 큰 시작 모델은 OpenAI의 GPT-2 1.3B 모델입니다).

## Before Get Started

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

의존성:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (GPT-2 체크포인트 로드용)
-  `datasets` for huggingface datasets <3 (OpenWebText를 다운로드하고 전처리하려면)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## Quick Start
딥러닝 전문가가 아니지만, GPT-2 의 놀라운 성능을 경험해보고 싶다면, 가장 빠른 방법은 셰익스피어 작품을 대상으로 문자 수준의 GPT를 훈련하는 것입니다. 먼저, 셰익스피어 텍스트를 단일 (1MB) 파일로 다운로드하고, 이를 큰 스트림의 정수로 변환합니다:

```sh
python data/shakespeare_char/prepare.py
```

이 작업은 `train.bin`과 `val.bin`을 해당 데이터 디렉터리에 생성합니다. 이제 GPT를 훈련할 시간입니다. 모델의 크기는 시스템의 계산 자원에 따라 다릅니다:

**GPU가 있습니다.** 좋습니다. [config/train_shakespeare_char.py](config/train_shakespeare_char.py) 설정 파일에 제공된 설정으로 아기 GPT를 빠르게 훈련할 수 있습니다:

```sh
python train.py config/train_shakespeare_char.py
```

이 설정 파일을 열어보면, 최대 256자 컨텍스트 크기, 384개의 특징 채널, 6층의 트랜스포머(각 층에 6개의 헤드)를 사용하는 GPT를 훈련 중인 것을 볼 수 있습니다. A100 GPU 하나로 이 훈련은 약 3분이 걸리며, 최고의 검증 손실은 1.4697입니다. 설정에 따라 모델 체크포인트는 `--out_dir` 디렉터리 `out-shakespeare-char`에 저장됩니다. 훈련이 완료되면 이 디렉터리를 가리키는 샘플링 스크립트를 사용하여 최상의 모델에서 샘플을 생성할 수 있습니다:

```sh
python sample.py --out_dir=out-shakespeare-char
```

몇 가지 샘플이 생성됩니다. 예를 들면:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

웃음 `¯\_(ツ)_/¯`. GPU에서 3분 동안 훈련한 문자 수준 모델치고는 나쁘지 않습니다. 더 나은 결과를 얻으려면, 이 데이터셋에서 사전 훈련된 GPT-2 모델을 미세 조정하는 것이 좋습니다(추후의 미세 조정 섹션 참조).

**맥북만 있습니다.** (또는 저렴한 컴퓨터) 걱정하지 마세요. GPT를 여전히 훈련할 수 있지만, 설정을 조금 낮추는 것이 좋습니다. 최신 PyTorch nightly를 설치하는 것이 좋습니다([여기서 선택](https://pytorch.org/get-started/locally/))하면 코드가 더 효율적으로 실행될 가능성이 큽니다. 하지만 그렇지 않더라도, 간단한 훈련은 다음과 같이 실행될 수 있습니다:

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

여기서, GPU 대신 CPU를 사용하기 때문에 `--device=cpu`를 설정하고 PyTorch 2.0 컴파일도 `--compile=False`로 비활성화해야 합니다. 그런 다음 평가할 때 조금 더 소음이 있지만 빠르게 추정 (`--eval_iters=20`, 이전 200회에서 감소)을 하고, 컨텍스트 크기를 256자 대신 64자로, 배치 크기를 64개 대신 12개의 예제로 설정합니다. 훨씬 작은 트랜스포머(4층, 4개의 헤드, 128 임베딩 크기)를 사용하고, 반복 횟수를 2000으로 줄입니다 (`--lr_decay_iters`). 네트워크가 작기 때문에 정규화를 완화합니다 (`--dropout=0.0`). 이 훈련은 여전히 약 3분 정도 걸리지만, 손실은 1.88로 줄어들고 샘플도 약간 악화됩니다. 그래도 재미있습니다:

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
다음과 같은 샘플을 생성합니다:

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

CPU에서 약 3분 동안 훈련한 결과로는 나쁘지 않습니다. 하이퍼파라미터를 조정하거나 네트워크 크기, 컨텍스트 길이(`--block_size`), 훈련 시간 등을 늘리면 더 나은 결과를 얻을 수 있습니다.

마지막으로, Apple Silicon MacBook과 최신 PyTorch 버전을 사용 중이라면 `--device=mps`(Metal Performance Shaders의 약어)를 추가하세요. 그러면 PyTorch가 칩에 내장된 GPU를 사용하여 훈련 속도를 2-3배까지 크게 향상시키고, 더 큰 네트워크를 사용할 수 있습니다. [Issue 28](https://github.com/karpathy/nanoGPT/issues/28)을 참조하세요.

## GPT-2 재현

보다 진지한 딥러닝 전문가는 GPT-2 결과를 재현하는 것에 관심이 있을 수 있습니다. 그래서 여기 있습니다 - 먼저, [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/)라는 OpenAI의 (비공개) WebText의 오픈 복제본을 사용하여 데이터셋을 토큰화합니다:

```sh
python data/openwebtext/prepare.py
```

이 명령은 [OpenWebText](https://huggingface.co/datasets/openwebtext) 데이터셋을 다운로드하고 토큰화합니다. GPT2 BPE 토큰 ID를 하나의 시퀀스로 저장한 `train.bin`과 `val.bin`을 생성합니다. 이제 훈련을 시작할 준비가 되었습니다. GPT-2 (124M)를 재현하려면 적어도 8XA100 40GB 노드가 필요하며 다음을 실행해야 합니다:

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

이 훈련은 약 4일 동안 PyTorch Distributed Data Parallel(DDP)을 사용하여 실행되며, 손실은 약 2.85로 감소합니다. 이제 GPT-2 모델은 OWT에서 평가할 때 검증 손실이 약 3.11이지만, 이를 미세 조정하면 약 2.85 영역으로 감소합니다 (도메인 차이 때문입니다), 두 모델이 거의 일치하게 됩니다.

클러스터 환경에서 여러 GPU 노드를 사용할 수 있다면, 예를 들어 2개의 노드를 사용하여 GPU를

 16개로 확장하는 경우:

```sh
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d train.py config/train_gpt2.py
```

또는, 미세 조정을 원한다면, 예를 들어, `train.py`를 일부 문서나 새로운 데이터에 대해 실행합니다:

```sh
python train.py config/finetune_gpt2.py --init_from=gpt2-xl --data_dir=data/my_custom_data
```

대규모 GPT-2 모델(GPT2-xl)을 특정 데이터를 사용하여 미세 조정한 후, `sample.py`를 사용하여 최종 모델을 샘플링할 수 있습니다. 미세 조정된 체크포인트에 가장 흥미로운 예시가 생성될 것입니다.

# Attractive Reproduction Guide

이 문서는 나중에 `attractive` 실험만 다시 재현할 때 그대로 따라 쓸 수 있는 실행 가이드다.

## 목표

- 태스크: `attractive`
- 모델: `alexnet`
- 방법: `van` baseline
- 모드: `eo`
- pretrained: `True`
- 반복 실험 수: `5`
- epoch 수: `5`
- 전체 데이터 사용

## 재현 설정

- `lr=1e-4`
- `eo_batch_size=128`
- `eval_batch_size=512`
- `train_num_workers=8`
- `num_workers=4`
- `AMP=False`
- `scratch=False`  -> pretrained 사용

## 결과 저장 위치

- 디렉터리: `results/celeba_attr_baseline_pretrained_full_ex5_ep5`
- 로그 파일: `results/celeba_attr_baseline_pretrained_full_ex5_ep5/celeba_attr_eo.txt`

## 실행 전 권장 사항

기존 CelebA 실험이 돌고 있으면 먼저 중지:

```bash
cd /workspace/RUNNER
pkill -f "python -u celeba_main.py|python .*celeba_main.py" || true
```

기존 attractive baseline 결과를 완전히 새로 시작하고 싶으면 로그 삭제:

```bash
cd /workspace/RUNNER
rm -f results/celeba_attr_baseline_pretrained_full_ex5_ep5/celeba_attr_eo.txt
```

## 재현 실행 명령

```bash
cd /workspace/RUNNER
python -u celeba_main.py \
  --task attractive \
  --model alexnet \
  --mode eo \
  --method van \
  --lr 1e-4 \
  --ex_num 5 \
  --epochs 5 \
  --eo_batch_size 128 \
  --eval_batch_size 512 \
  --train_num_workers 8 \
  --num_workers 4 \
  --log_interval 100 \
  --no-amp \
  --results_dir results/celeba_attr_baseline_pretrained_full_ex5_ep5
```

## 진행 상황 확인

```bash
cd /workspace/RUNNER
tail -f results/celeba_attr_baseline_pretrained_full_ex5_ep5/celeba_attr_eo.txt
```

## 로그 해석

- `Seed set to N`: 현재 `N`번째 seed 실행 시작
- `Epoch: K`: 현재 epoch
- `step a/b`: 현재 epoch 내 학습 step 진행률
- `idx`: 해당 seed에서 validation gap이 가장 좋았던 epoch 번호(1부터 시작)
- 마지막 `Average Precision`, `eo gap`: 5개 seed 평균 결과

## 참고

- 현재 코드베이스는 모델 checkpoint를 자동 저장하지 않고, 결과 로그 txt만 저장한다.
- 따라서 재현 후 남는 산출물은 기본적으로 `celeba_attr_eo.txt` 로그 파일이다.
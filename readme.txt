# 가상환경 on -> conda activate herok97
# train_mean_scale_hyperprior.py -> 학습용
# test_mean_scale_hyperprior.py -> 테스트용
# custom_model.py -> 모델 수정
# load_model.py -> 신경 쓸 필요 X

1. train

    1) CUDA_VISIBLE_DEVICES={your_gpu_number} python train_mean_scale_hyperprior.py --quality {1-8}

    2) --quality가 높을수록 lambda가 높음

2. test (약 20초 소요)

    1) CUDA_VISIBLE_DEVICES={your_gpu_number} python test_mean_scale_hyperprior.py --quality {1-8}
               --checkpoint {your model dir}
    2) --checkpoint 값을 넣어주지 않으면 pre-trained 모델 사용

3. validation / save model / logging

    1) training 진행 시 validation은 기본적으로 5000 step 마다 수행하며 log 폴더에 기록 (tensorboard에 loss 기록)

    2) 모델은 save 폴더에 저장되며 10만 step 마다 저장되게 되어있음, 또한, best 모델을 매번 함께 저장

4. scheduling

    1) 학습은 최대 500만 step을 수행하는데, scheduler에 의해 초기 learning rate 1e-4에서 loss가 내려가지 않을 때마다 0.5을 곱하게 됨.

    2) 학습은 항상 500만 step을 모두 수행하는 것이 아닌, learning rate가 1e-6 보다 작아지면 학습이 중지되도록 설정되어있음.

5. Dataset

    1) 학습에 사용할 데이터셋은 Vimeo90K로 비디오 데이터셋이지만, 이미지 학습에도 사용됨 (448x256)

    2) 기본적으로 256x256 crop을 하도록 transform을 정의해놨으므로, 이 부분을 수정해야 한다면 수정해도 됨

6. Model customizing

    1) custom_model.py는 compressai에서 제공하는 MeanScaleHyperprior 모델을 상속받은 모델로 forward는 학습때 사용하고
        실제 테스트 시점에서 압축 및 복원은 진행할 때는 compress와 decompress 함수를 이용함

    2) custom_model.py 내의 CustomMeanScaleHyperprior 클래스 내의 함수를 수정하여 진행하거나 train/test 코드를 수정하여 실험 진행

7. Configuration
    "-q", "--quality", type=int, default=0, help="quality of the model"
    "-save_dir", "--save_dir", type=str, default='save/', help="save_dir"
    "-log_dir", "--log_dir", type=str, default='log/', help="log_dir"
    "-total_step", default=5000000, type=int, help="total_step (default: %(default)s)"
    "-test_step", "--test_step", default=5000,
    "-save_step", default=100000,
    "-lr", "--learning-rate", default=1e-4,
    "-n", "--num-workers", type=int, default=4,
    "--patch-size", default=(256, 256),
    "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    "--test-batch-size", default=1, help="Test batch size (default: %(default)s)",
    "--aux-learning-rate", default=1e-3, help="Auxiliary loss learning rate (default: %(default)s)",
    "--checkpoint", type=str, help="Path to a checkpoint"

import torch

print(torch.cuda.is_available())  # CUDA 사용 가능 여부 확인
print(torch.cuda.device_count())  # 시스템에 있는 GPU 수 확인
print(torch.cuda.get_device_name(0))  # 첫 번째 GPU의 이름 확인
print(torch.cuda.current_device())  # 현재 사용 중인 GPU의 인덱스 확인
torch.cuda.memory_allocated()  # 현재 할당된 GPU 메모리 확인
torch.cuda.memory_cached()  # 현재 캐시된 GPU 메모리 확인

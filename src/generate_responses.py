import torch
print(torch.cuda.is_available())  # Должно вернуть True
print(torch.cuda.device_count())  # Показывает количество доступных GPU
print(torch.cuda.get_device_name(0))  # Показывает название устройства

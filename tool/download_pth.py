import os
import shutil
import urllib.request

# 下載模型
url = 'https://github.com/bubbliiiing/unet-pytorch/releases/download/v1.0/unet_resnet_voc.pth'
urllib.request.urlretrieve(url, 'unet_resnet_voc.pth')
pth_dir = r'C:\project\unet-pytorch\pth_folder'
os.makedirs(pth_dir, exist_ok=True)
# 複製檔案
shutil.copy('unet_resnet_voc.pth', pth_dir)

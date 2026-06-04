cd C:/project/docker_pytorch
# 建置
docker build --no-cache -t unet-train:headless .
# 執行
docker run --gpus all --rm -it  unet-train:headless /bin/bash   

docker run --gpus all --rm -it -v C:\project\unet-pytorch:/workspace/unet-pytorch unet-train:headless /bin/bash               

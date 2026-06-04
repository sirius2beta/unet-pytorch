# 建置
docker build --no-cache -t unet-train:headless .

# 執行
docker run --gpus all --rm -it  unet-train:headless /bin/bash   

# 執行 掛載資料夾
docker run --gpus all --rm -it -v C:\project\unet-pytorch:/workspace/unet-pytorch unet-train:headless /bin/bash               

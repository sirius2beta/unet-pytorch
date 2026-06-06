# predict_single.py (放在你指定的 project_dir 裡面)
import sys
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from unet import Unet  # 確保同資料夾下有 unet.py 且路徑可以被 import

SEAGRASS_CLASS = 1

def compute_coverage(pr_mask):
    return np.sum(pr_mask == SEAGRASS_CLASS) / pr_mask.size

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Error: 必須提供圖片路徑與模型路徑 (e.g., script.py <img_path> <model_path>)")
    
    img_path = sys.argv[1]
    model_path = sys.argv[2]
    
    try:
        # 這裡假設你的 Unet 類別可以接受 model_path 參數來初始化權重
        # 你可能需要稍微修改你的 unet.py 讓他支援動態傳入 model_path
        unet = Unet(model_path=model_path) 
        
        image = Image.open(img_path)
        _, pred_mask = unet.detect_image(image, return_mask=True)
        coverage = compute_coverage(pred_mask)
        
        # 僅印出 coverage 數值，讓 UI 擷取
        print(f"{coverage:.6f}") 
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
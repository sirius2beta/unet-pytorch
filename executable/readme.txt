pip install pyinstaller

pyinstaller --noconsole --onefile maskView.py
pyinstaller --noconsole --onefile exportDataset.py

cd D:\project\unet-pytorch\tool
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
pyinstaller --noconsole --onefile maskView.py
# detBlendPipe
Data generator and Object Detection using BlenderProc and Detectron2

# Installation

Use venv if you need it.
```
pip install -r requirements.txt
git clone https://github.com/DLR-RM/BlenderProc.git

# skip this if you dont want sample haven dataset
cd BlenderProc
python scripts/download_haven.py 
```

This will install detectron2 and blenderproc.

# Usage
```
streamlit run front_end/main.py
```

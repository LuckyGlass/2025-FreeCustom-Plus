git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
cd Grounded-Segment-Anything
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install --upgrade diffusers[torch]
git submodule update --init --recursive
cd grounded-sam-osx && bash install.sh
git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
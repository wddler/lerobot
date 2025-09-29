# Activate your lerobot environment
conda activate lerobot

# Find where the lerobot module is located
python -c "import lerobot; print(lerobot.__file__)"

# Find where the cv2 module is located within that environment
python -c "import cv2; print(cv2.__file__)"

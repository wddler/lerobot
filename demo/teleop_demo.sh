#!/bin/bash
export HOME=/home/robohouse
export PATH="$HOME/miniconda3/bin:$PATH"

# Ensure conda is available
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Activate conda environment
conda activate lerobot

# Debug: Check if lerobot is installed in the correct environment
python -c "import sys; print(sys.executable)"
python -c "import lerobot"  # This will fail if the package isn't installed


# Change directory and run the Python script
cd "$HOME/lerobot"
exec lerobot-teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=false

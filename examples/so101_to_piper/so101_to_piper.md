### Teleoperation: 

1. Make sure the right CAN is activated.

2. run `uv run python examples/so101_to_piper/teleoperate.py`

or

lerobot-teleoperate \
  --robot.type=piper_follower \
  --robot.can_port=can0 \
  --robot.id=my_piperx \
  --teleop.type=so101_leader_ee \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_leader_arm_ee.json \
  --teleop.urdf_path=<path_to_so101_urdf> \
  --teleop.scale_factor=2.0
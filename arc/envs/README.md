# Env lock files
`train.txt` and `gen.txt` are written by `arc/setup_envs.sh` on B's account (task B1) and committed here.
A and C build from them: `CANARY_PID=<pid> bash arc/setup_envs.sh all --from-lock`. Change them only by PR + board post.

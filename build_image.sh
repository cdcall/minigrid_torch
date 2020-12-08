#!/usr/bin/env bash

git clone https://github.com/drucej/minigrid_torch.git
git checkout dev

docker build -t rl_sandbox .
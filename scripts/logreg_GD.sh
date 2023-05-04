#!/bin/bash
python -B ../main.py --dataset=mnist --model=logreg --lr=0.5 --optimizer=GD --augment=True --gamma=0.1
          
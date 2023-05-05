#!/bin/bash
python -B ../main.py --dataset=mnist --model=logreg --lr=0.2 --optimizer=Newton --augment=True --gamma=0.1
          
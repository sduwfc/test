#!/bin/bash


	# --video_path=../../basicvsr/city.mp4 # (144, 176, 3)


python3 evaluate.py \
	../../dataset/basicvsr/BDx4 \
	../../dataset/basicvsr/GT \
	--spynet-mode=../../spynet_1684_f32/compilation.bmodel \
	--forward-residual-model=../../forward_1684/compilation.bmodel \
	--backward-residual-model=../../backward_1684_int8/compilation.bmodel \
	--upsample-model=../../upsample_1684/compilation.bmodel

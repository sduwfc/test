export LD_LIBRARY_PATH=/workspace/tpu-nntc/lib/:$LD_LIBRARY_PATH

python3 inference.py \
	--spynet-model=/workspace/model_basicvsr/spynet_workspace/basicvsr_spynet_f32.bmodel \
	--forward-residual-model=/workspace/model_basicvsr/forward_workspace/basicvsr_forward_f32.bmodel \
	--backward-residual-model=/workspace/model_basicvsr/backward_workspace/basicvsr_backward_f32.bmodel \
	--upsample-model=/workspace/model_basicvsr/upsample_workspace/basicvsr_upsample_f32.bmodel \
	--dump-npz \
	--dump-lmdb \
	--imagedir=../../dataset/basicvsr/eval/BDx4/calendar 


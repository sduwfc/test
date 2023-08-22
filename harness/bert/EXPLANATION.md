# Bert Benchmark 在 model zoo 环境下运行说明

-----------------------------------


## Inference

```shell
python3 run.py --model=/path/to/bmodel \
                --data=/path/to/dev-v1.1.json \
                --count=100 \
                --accuracy \

```

参数说明：

--model: bmodel路径

--data: 验证集数据路径

--accuracy: 是否打开计算精度

--count: 运行的样本数，计算精度时最多为10833（数据集长度）

--devices: 使用的设备id，例如 --devices 0 1 2

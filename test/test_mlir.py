import logging
import pytest
from utils import container_run

@pytest.mark.build
@pytest.mark.mlir
@pytest.mark.parametrize('target', ['BM1684', 'BM1684X'])
def test_mlir_efficiency(target, mlir_env, get_val_dataset):
    if not mlir_env['case_list']:
        logging.info(f'Skip efficiency test')
        return

    container_run(mlir_env, f'python3 -m tpu_perf.build {mlir_env["case_list"]} \
        --outdir mlir_out_{target} --target {target} --mlir --report mlir_{target}_acc_cases_status.json')


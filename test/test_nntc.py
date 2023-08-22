import logging
import pytest
from utils import container_run

@pytest.fixture(scope='module')
def make_lmdb(nntc_env):
    if not nntc_env['case_list']:
        logging.info(f'Skip nntc make lmdb')
        return
    cmd = 'pip3 install -r /workspace/requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple'
    container = nntc_env['container']
    logging.info(cmd)

    for i in range(4):
        try:
            ret, output = container.exec_run(f'bash -c "{cmd}"', tty=True)
            break
        except:
            if i == 0:
                logging.info(f'Try pip3 install -r /workspace/requirements.txt a {i}th time!')
            elif i == 3:
                output = output.decode()
                logging.error(f'------>\n{output}')
                assert ret == 0

    container_run(nntc_env, f'python3 -m tpu_perf.make_lmdb {nntc_env["case_list"]}')

@pytest.mark.build
@pytest.mark.nntc
@pytest.mark.parametrize('target', ['BM1684', 'BM1684X'])
def test_nntc_efficiency(target, nntc_env, get_val_dataset):
    if not nntc_env['case_list']:
        logging.info(f'Skip nntc efficiency test')
        return

    # Build for efficiency test
    container_run(nntc_env, f'python3 -m tpu_perf.build --time {nntc_env["case_list"]} \
        --outdir out_eff_{target} --target {target} --report nntc_{target}_eff_cases_status.json')

@pytest.mark.build
@pytest.mark.nntc
@pytest.mark.parametrize('target', ['BM1684', 'BM1684X'])
def test_nntc_accuracy(target, nntc_env, get_val_dataset, make_lmdb):
    if not nntc_env['case_list']:
        logging.info(f'Skip nntc accuracy test')
        return

    # Build for accuracy test
    container_run(nntc_env, f'python3 -m tpu_perf.build {nntc_env["case_list"]} \
        --outdir out_acc_{target} --target {target} --report nntc_{target}_acc_cases_status.json')
import pytest
import logging
import docker
import time
import os
import uuid

import re
import io
import tarfile
import requests
from ftplib import FTP
import subprocess
import utils

class FTPClient:
    """
    ftp://user_name:password@hostname
    """
    def __init__(self, url):
        prog = re.compile('ftp://(.+)@')
        pat = prog.search(url)
        if pat:
            self.user, self.passwd = pat.group(1).split(':')
        else:
            self.user, self.passwd = None, None
        self.host = prog.sub('', url)

        self.release_type = 'daily_build'
        if os.environ.get('GITHUB_REF', '').endswith('stable'):
            self.release_type = 'release_build'

        self.session = FTP(self.host, user=self.user, passwd=self.passwd)

    def download_and_untar(self, fn):
        logging.info(f'Download & extract {fn}')
        buf = io.BytesIO()
        self.session.retrbinary(
            f'RETR {fn}',
            buf.write)
        buf.seek(0)
        tar = tarfile.open(fileobj=buf)
        tar.extractall()

    def download(self, fn, out_dir):
        out_fn = os.path.join(out_dir, os.path.basename(fn))
        logging.info(f'Download {fn} to {out_fn}')
        with open(out_fn, 'wb') as fp:
            self.session.retrbinary(
                f'RETR {fn}',
                fp.write)
        return out_fn

    def get_release(self, name, get_fn=None, is_tar=False,is_dev=False):
        if get_fn is None:
            get_fn = lambda x: x.startswith(f'{name}_')
        path = f'/sophon-sdk/{name}/{self.release_type}/latest_release'
        self.session.cwd(path)  # current working directory
        fn = next(filter(get_fn, self.session.nlst())) # filename
        logging.info(f'Latest {name} package is {fn}')
        if is_tar:
            out_dir = fn.replace('.tar.gz', '')
            if os.path.exists(out_dir):
                logging.info(f'{out_dir} already exists')
                return out_dir
            self.download_and_untar(os.path.join(path, fn))
            return out_dir
        else:
            self.download(os.path.join(path, fn), '.')
            return fn

    def get_tar(self, name):
        out_dir = self.get_release(name, is_tar=True)
        for m in glob.glob(f'{name}*'):
            if m != out_dir:
                remove_tree(m)
        return out_dir

    def get_nntc(self):
        return self.get_tar('tpu-nntc')

    def get_mlir(self):
        return self.get_tar('tpu-mlir')

    def get_libsophon(self):
        return self.get_release(
            'libsophon',
            get_fn=lambda x: x.startswith('sophon-libsophon_') and x.endswith('amd64.deb'), is_dev=True)

from html.parser import HTMLParser

class ReleasePageParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        super(ReleasePageParser, self).__init__(*args, **kwargs)
        self.results = []

    def handle_starttag(self, tag, attrs):
        if tag == 'include-fragment':
            attrs = dict(attrs)
            m = re.match('^.+(\\d+\\.)+\\d+$', attrs.get('src', ''))
            if not m:
                return
            self.results.append(m.group(0))

class ExpandParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        super(ExpandParser, self).__init__(*args, **kwargs)
        self.results = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            attrs = dict(attrs)
            self.results.append(attrs.get('href'))

def get_latest_tpu_perf():
    backoff = 0.5
    url = 'https://github.com/sophgo/tpu-perf/releases'
    for i in range(10):
        try:
            resp = requests.get(url, timeout=15)
            break
        except requests.exceptions.Timeout:
            logging.warning(f'Failed to query {url}, retry after {backoff}s')
            time.sleep(backoff)
            backoff *= 2
    assert resp

    resp.raise_for_status()
    parser = ReleasePageParser()
    parser.feed(resp.text)

    page = parser.results[0]
    backoff = 0.5
    for i in range(10):
        try:
            resp = requests.get(page, timeout=15)
            break
        except requests.exceptions.Timeout:
            logging.warning(f'Failed to query {page}, retry after {backoff}s')
            time.sleep(backoff)
            backoff *= 2
    assert resp

    resp.raise_for_status()
    parser = ExpandParser()
    parser.feed(resp.text)

    return parser.results

tpu_perf_whl = None
@pytest.fixture(scope='session')
def latest_tpu_perf_whl():
    import platform
    arch = platform.machine()
    global tpu_perf_whl
    if not tpu_perf_whl:
        tpu_perf_whl = next(filter(lambda x: arch in x, get_latest_tpu_perf()))
    return f'https://github.com/{tpu_perf_whl}'

import shutil
import glob
def remove_tree(path):
    for m in glob.glob(path):
        logging.info(f'Removing {m}')
        if os.path.isdir(m):
            shutil.rmtree(m)
        else:
            os.remove(m)

dummy_github_output = '/tmp/dummy.github.output.txt'
def read_github_output(key):
    if 'GITHUB_OUTPUT' in os.environ:
        return os.environ[key]
    else:
        with open(dummy_github_output) as f:
            data = dict(
                line.strip(' \n').split('=')
                for line in f.readlines() if line)
            return data[key]

def write_github_output(key, value):
    if 'GITHUB_OUTPUT' in os.environ:
        mode = 'a'
        output_fn = os.environ['GITHUB_OUTPUT']
        logging.info(f'Writing {key} to GITHUB_OUTPUT')
    else:
        mode = 'w'
        output_fn = dummy_github_output

    with open(output_fn, mode) as f:
        f.write(f'{key}={value}\n')

def get_toolchain_cmtId(sdk):
    toolchain = sdk.split('_')[0]

    if toolchain == 'tpu-mlir':
        cmt_id = sdk.split('_')[1].split('-')[1][1:]
    elif toolchain == 'tpu-nntc':
        toolchain = 'nntoolchain'
        cmt_id = sdk.split('_')[1].split('-')[1]

    output = f'{toolchain}_{cmt_id}'
    return output

@pytest.fixture(scope='session')
def nntc_docker(latest_tpu_perf_whl):
    # Env assertion
    assert os.path.exists('/run/docker.sock')

    root = os.path.dirname(os.path.dirname(__file__))
    logging.info(f'Working dir {root}')
    os.chdir(root)

    # Download
    ftp_server = os.environ.get('FTP_SERVER')
    assert ftp_server
    f = FTPClient(ftp_server)
    nntc_dir = f.get_nntc()

    # Docker init
    client = docker.from_env(timeout=360)
    image = 'sophgo/tpuc_dev:v2.1'
    client.images.pull(image)

    # Glob kernel module
    import glob
    kernel_module = glob.glob(os.path.join(nntc_dir, 'lib/*kernel_module*'))
    assert kernel_module
    kernel_module = kernel_module[0]

    # NNTC container
    nntc_container = client.containers.run(
        image, 'bash',
        volumes=[f'{root}:/workspace'],
        restart_policy={'Name': 'always'},
        environment=[
            f'PATH=/workspace/{nntc_dir}/bin:/usr/local/bin:/usr/bin:/bin',
            f'BMCOMPILER_KERNEL_MODULE_PATH=/workspace/{kernel_module}',
            f'NNTC_TOP=/workspace/{nntc_dir}'],
        tty=True, detach=True)

    if 'GITHUB_ENV' in os.environ:
        with open(os.environ['GITHUB_ENV'], 'a') as f:
            f.write(f'NNTC_CONTAINER={nntc_container.name}\n')

    # Remove old outputs
    nntc_container.exec_run(f'bash -c "rm -rf *out* *_cases_status.json"', tty=True)

    logging.info(f'Setting up NNTC')
    ret, _ = nntc_container.exec_run(
        f'bash -c "source /workspace/{nntc_dir}/scripts/envsetup.sh"',
        tty=True)
    assert ret == 0

    logging.info(f'NNTC container {nntc_container.name}')

    yield dict(docker=client, container=nntc_container)

    # Chown so we can delete them later
    dirs_to_remove = ['*.tar', '*out*', 'data', '.tar.gz']
    nntc_container.exec_run(
        f'bash -c "chown -R {os.getuid()} {" ".join(dirs_to_remove)}"',
        tty=True)

    # Pack bmodels for runtime jobs
    toolchain_version = get_toolchain_cmtId(nntc_dir)
    model_tar = f'{toolchain_version}_{uuid.uuid4()}.tar'
    for target in ['BM1684', 'BM1684X']:
        upload_bmodel(target, model_tar, f'<(find out*_{target} -name \'*.compilation\' 2>/dev/null)')
    write_github_output('NNTC_MODEL_TAR', model_tar)

    # Docker cleanup
    logging.info(f'Removing NNTC container {nntc_container.name}')
    nntc_container.remove(v=True, force=True)

    for d in dirs_to_remove:
        remove_tree(d)

@pytest.fixture(scope='session')
def mlir_docker(latest_tpu_perf_whl):
    # Env assertion
    assert os.path.exists('/run/docker.sock')

    root = os.path.dirname(os.path.dirname(__file__))
    logging.info(f'Working dir {root}')
    os.chdir(root)

    # Download
    ftp_server = os.environ.get('FTP_SERVER')
    assert ftp_server
    f = FTPClient(ftp_server)
    mlir_dir = f.get_mlir()

    # Docker init
    image = 'sophgo/tpuc_dev:latest'
    client = docker.from_env(timeout=360)
    logging.info(f'Pull image {image}')
    client.images.pull(image)

    # MLIR container
    logging.info(f'Setting up MLIR')
    mlir_container = client.containers.run(
        image, 'bash',
        volumes=[f'{root}:/workspace'],
        restart_policy={'Name': 'always'},
        environment=[
            f'TPUC_ROOT=/workspace/{mlir_dir}',
            f'PATH=/workspace/{mlir_dir}/bin:' \
            f'/workspace/{mlir_dir}/python/tools:' \
            f'/workspace/{mlir_dir}/python/utils:' \
            f'/workspace/{mlir_dir}/python/test:' \
            f'/workspace/{mlir_dir}/python/samples:' \
            f'/usr/local/bin:/usr/bin:/bin',
            f'LD_LIBRARY_PATH=/workspace/{mlir_dir}/lib',
            f'PYTHONPATH=/workspace/{mlir_dir}/python:' \
            f'/usr/local/python_packages/mlir_core:' \
            f'/usr/local/python_packages/:'],
        tty=True, detach=True)

    # For cleanup jobs in case we fail
    if 'GITHUB_ENV' in os.environ:
        with open(os.environ['GITHUB_ENV'], 'a') as f:
            f.write(f'MLIR_CONTAINER={mlir_container.name}\n')

    logging.info(f'MLIR container {mlir_container.name}')

    # Remove old outputs
    mlir_container.exec_run(f'bash -c "rm -rf *out* *_cases_status.json"', tty=True)

    yield dict(docker=client, container=mlir_container)

    # Pack bmodels for runtime jobs
    toolchain_version = get_toolchain_cmtId(mlir_dir)
    model_tar = f'{toolchain_version}_{uuid.uuid4()}.tar'
    for target in ['BM1684', 'BM1684X']:
        relative_fns = set()
        for dirpath, dirnames, filenames in os.walk(f'mlir_out_{target}'):
            for fn in filenames:
                if fn.endswith('compilation.bmodel'):
                    continue
                if fn.endswith('.bmodel') or fn.endswith('profile_0.txt'):
                    relative_fns.add(os.path.join(dirpath, fn))
                if fn.endswith('.dat'):
                    relative_fns.add(dirpath)
        list_fn = 'out_list.txt'
        with open(list_fn, 'w') as f:
            f.write('\n'.join(relative_fns))
        upload_bmodel(target, model_tar, list_fn)
    write_github_output('MLIR_MODEL_TAR', model_tar)
    
    # Chown so we can delete them later
    dirs_to_remove = ['*.tar', '*out*', 'data', '*list.txt', '*.tar.gz']
    mlir_container.exec_run(
        f'bash -c "chown -R {os.getuid()} {" ".join(dirs_to_remove)}"',
        tty=True)

    # Docker cleanup
    logging.info(f'Removing MLIR container {mlir_container.name}')
    mlir_container.remove(v=True, force=True)

    for d in dirs_to_remove:
        remove_tree(d)

def git_commit_id(rev):
    p = subprocess.run(
        f'git rev-parse {rev}',
        shell=True, check=True,
        capture_output=True)
    return p.stdout.decode().strip(' \n')

def git_commit_parents(rev='HEAD'):
    p = subprocess.run(
        f'git rev-parse {rev}^@',
        shell=True, check=True,
        capture_output=True)
    return p.stdout.decode().strip(' \n').split()

def dig(c, callback, depth=0, max_depth=100):
    if not callback(c):
        return
    if depth >= max_depth:
        return
    for p in git_commit_parents(c):
        dig(p, callback, depth + 1, max_depth)

def get_relevant_commits():
    head_parents = git_commit_parents()
    if len(head_parents) == 1:
        return ['HEAD']
    assert len(head_parents) == 2

    base_set = set()
    def cb(x):
        if x in base_set:
            return False
        base_set.add(x)
        return True
    dig(git_commit_id('origin/main'), cb)

    ps = [p for p in head_parents if p not in base_set]
    result = []
    while ps:
        result += ps
        new_ps = []
        for p in ps:
            new_ps += [new_p for new_p in git_commit_parents(p) if new_p not in base_set]
        ps = new_ps

    return result

def git_changed_files(rev):
    p = subprocess.run(
        f'git show --pretty="" --diff-filter=ACMRTUXB --name-only {rev}',
        shell=True, check=True,
        capture_output=True)
    return p.stdout.decode().strip(' \n').split()

from functools import reduce
@pytest.fixture(scope='session')
def case_list():
    if 'TEST_CASES' in os.environ:
        return os.environ['TEST_CASES'].strip() or '--full'

    if os.environ.get('FULL_TEST'):
        return '--full'

    files = reduce(
        lambda acc, x: acc + x,
        [git_changed_files(c) for c in get_relevant_commits()], [])

    # Skip certain files
    files = [
        f for f in files
        if not f.endswith('.md')
        and not f.endswith('.txt')
        and not os.path.basename(f).startswith('.')]

    is_model = lambda x: x.startswith('vision') or x.startswith('language')
    files = [f for f in files if is_model(f)]

    dirs = set([os.path.dirname(f) for f in files])
    def has_config(d):
        try:
            next(filter(lambda x: x.endswith('config.yaml'), os.listdir(d)))
        except StopIteration:
            return False
        else:
            return True
    s = ' '.join(d for d in dirs if has_config(d))
    return s

@pytest.fixture(scope='session')
def nntc_env(nntc_docker, latest_tpu_perf_whl, case_list):
    ret, _ = nntc_docker['container'].exec_run(
        f'bash -c "pip3 install {latest_tpu_perf_whl}"',
        tty=True)
    assert ret == 0

    logging.info(f'Running cases "{case_list}"')

    yield dict(**nntc_docker, case_list=case_list)

@pytest.fixture(scope='session')
def mlir_env(mlir_docker, latest_tpu_perf_whl, case_list):
    ret, _ = mlir_docker['container'].exec_run(
        f'bash -c "pip3 install {latest_tpu_perf_whl}"',
        tty=True)
    assert ret == 0

    logging.info(f'Running cases "{case_list}"')

    yield dict(**mlir_docker, case_list=case_list)

def pytest_addoption(parser):
    parser.addoption('--target', action='store')

@pytest.fixture(scope='session')
def target(request):
    return request.config.getoption('--target')

import pandas as pd
def check_output_csv():
    csv_fns = glob.glob('*out*/*.csv')
    if len(csv_fns) == 0:
        logging.info('No .csv found!')
    else:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        logging.info(f'Number of csvs: {len(csv_fns)}')
        for fn in csv_fns:
            runtime_out = pd.read_csv(fn, encoding='utf-8', header=0)
            logging.info(f'{fn}\n{runtime_out}')

def download_bmodel(target, model_tar):
    assert model_tar, 'Model tar is empty'
    assert target, 'Please specify --target'
    assert 'SWAP_SERVER' in os.environ, 'SWAP_SERVER required'
    swap_server = os.environ['SWAP_SERVER']

    output_fn = f'{target}_{model_tar}'
    logging.info(f'Downloading {output_fn}')
    url = os.path.join(swap_server, output_fn)
    cmd = f'curl -s {url} | tar -x'
    execute_cmd(cmd)

def upload_bmodel(target, model_tar, T):
    fn = f'{target}_{model_tar}'
    assert 'SWAP_SERVER' in os.environ, 'SWAP_SERVER required'
    swap_server = os.environ['SWAP_SERVER']
    logging.info(f'Uploading {fn}')
    dst = os.path.join(swap_server, fn)
    subprocess.run(
        f'bash -c "tar -T {T} -cO | curl -s --fail {dst} -T - > /dev/null"',
        shell=True, check=True)

def package_csv(target, model_tar):
    import platform
    arch = platform.machine()

    fn = f'csv_{target}_ARM64_{model_tar}.gz' if arch == 'aarch64' else f'csv_{target}_X64_{model_tar}.gz'
    subprocess.run(
        f'bash -c "tar -T <(find ./ -name "*.csv") -cvf {fn}"',
        shell=True, check=True)

@pytest.fixture(scope='session')
def runtime_dependencies(latest_tpu_perf_whl):
    execute_cmd(f'pip3 install {latest_tpu_perf_whl} > /dev/null')

@pytest.fixture(scope='session')
def precision_dependencies(latest_tpu_perf_whl):
    execute_cmd(f'pip3 install {latest_tpu_perf_whl} > /dev/null')
    execute_cmd('pip3 install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple > /dev/null')

@pytest.fixture(scope='session')
def mlir_runtime(target, case_list):
    dirs_to_remove = ['*.tar', '*out*', 'data', '*.tar.gz']
    for d in dirs_to_remove:
        remove_tree(d)

    model_tar = read_github_output('MLIR_MODEL_TAR')
    assert model_tar, 'Model tar is empty'
    download_bmodel(target, model_tar)
    logging.info(f'Running cases "{case_list}"')

    yield dict(case_list=case_list)

    check_output_csv()
    package_csv(target, model_tar)

    # Cleanup
    for d in dirs_to_remove:
        remove_tree(d)

@pytest.fixture(scope='session')
def nntc_runtime(target, case_list):
    dirs_to_remove = ['*.tar', '*out*', 'data', '*.tar.gz']
    for d in dirs_to_remove:
        remove_tree(d)

    model_tar = read_github_output('NNTC_MODEL_TAR')
    assert model_tar, 'Model tar is empty'
    download_bmodel(target, model_tar)
    logging.info(f'Running cases "{case_list}"')

    yield dict(case_list=case_list)

    check_output_csv()
    package_csv(target, model_tar)

    # Cleanup
    for d in dirs_to_remove:
        remove_tree(d)

def execute_cmd(cmd):
    logging.info(cmd)
    ret = os.system(cmd)
    assert ret == 0, f'{cmd} failed!'

@pytest.fixture(scope='session')
def get_cifar100():
    data_server = os.environ.get('DATA_SERVER')
    assert data_server
    fn = 'cifar-100-python.tar.gz'

    if len(os.listdir('dataset/CIFAR100/cifar-100-python/')) >= 5:
        logging.info(f'{fn} already downloaded')
    else:
        url = os.path.join(data_server, fn)
        logging.info(f'Downloading {fn}')
        cmd = f'curl -s {url} | tar -zx --strip-components=1 ' \
             '-C dataset/CIFAR100/cifar-100-python/'
        execute_cmd(cmd)

@pytest.fixture(scope='session')
def get_imagenet_val():
    data_server = os.environ.get('DATA_SERVER')
    assert data_server
    fn = 'ILSVRC2012_img_val.tar'
    url = os.path.join(data_server, fn)
    dst = 'dataset/ILSVRC2012/ILSVRC2012_img_val/'
    if len(os.listdir(dst)) >= 50000:
        logging.info(f'{fn} already downloaded')
        return
    logging.info(f'Downloading {fn}')
    cmd = f'curl -s {url} | tar -x -C {dst}'
    execute_cmd(cmd)

@pytest.fixture(scope='session')
def get_coco2017_val():
    data_server = os.environ.get('DATA_SERVER')
    assert data_server

    fn = 'val2017.zip'
    url = os.path.join(data_server, fn)
    if len(os.listdir('dataset/COCO2017/val2017')) >= 5000:
        logging.info(f'{fn} already downloaded')
    else:
        logging.info(f'Downloading {fn}')
        cmd = f'curl -o val2017.zip -s {url}'
        execute_cmd(cmd)
        cmd = 'unzip -o val2017.zip -d dataset/COCO2017'
        execute_cmd(cmd)
        cmd = 'rm val2017.zip'
        execute_cmd(cmd)

    fn = 'annotations_trainval2017.zip'
    if len(os.listdir('dataset/COCO2017/annotations')) >= 7:
        logging.info(f'{fn} already downloaded')
    else:
        url = os.path.join(data_server, fn)
        logging.info(f'Downloading {fn}')
        cmd = f'curl -o annotations.zip -s {url}'
        execute_cmd(cmd)
        cmd = 'unzip -o annotations.zip -d dataset/COCO2017/'
        execute_cmd(cmd)
        cmd = 'rm annotations.zip'
        execute_cmd(cmd)

@pytest.fixture(scope='session')
def get_basicvsr_val():
    data_server = os.environ.get('DATA_SERVER')
    assert data_server

    fn = 'basicvsr_val.zip'
    url = os.path.join(data_server, fn)
    if len(os.listdir('dataset/basicvsr/eval')) >= 3:
        logging.info(f'{fn} already downloaded')
    else:
        logging.info(f'Downloading {fn}')
        cmd = f'curl -o {fn} -s {url}'
        execute_cmd(cmd)
        cmd = f'unzip -o {fn} -d dataset/basicvsr'
        execute_cmd(cmd)
        cmd = f'rm {fn}'
        execute_cmd(cmd)

@pytest.fixture(scope='session')
def get_val_dataset(get_imagenet_val, get_cifar100, get_coco2017_val, get_basicvsr_val):
    return 0

def main():
    logging.basicConfig(level=logging.INFO)

    files = reduce(
        lambda acc, x: acc + x,
        [git_changed_files(c) for c in get_relevant_commits()], [])
    print(files)

if __name__ == '__main__':
    main()

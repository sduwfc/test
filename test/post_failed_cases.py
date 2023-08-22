import logging
import os
import json
import glob
import requests
import time
import sys

def case_name_to_path(case_names):
    with open('./README.md', 'r') as f:
        for line in f.readlines():
            for i in range(len(case_names)):
                case_name = case_names[i].replace('_', '\_')
                if case_name in line:
                    case_names[i] = line.split('|')[2].split('(')[1].split(')')[0]
    return set(case_names)

def get_job_url(queue_id, auth):
    time.sleep(3)
    for i in range(20):
        que_res = requests.get(f'http://172.28.142.24:8092/queue/item/{queue_id}/api/json', auth=auth)
        parsed_data = json.loads(que_res.text)
        if 'executable' in parsed_data:
            res_url = parsed_data['executable']['url']
            return res_url
        time.sleep(1)
    return ""

def post_jenkins(json_param):
    jen_url = os.environ.get('JENKINS_URL')
    jen_job = '/job/model-zoo-regression'
    jen_param_url = jen_url + jen_job + '/buildWithParameters'
    auth_info = os.environ.get('JENKINS_AUTH').split(':')
    auth = (auth_info[0], auth_info[1])

    max_retries = 4
    retries = 1

    while retries < max_retries:
        try:
            response = requests.post(jen_param_url, auth=auth, params=json_param)
            if retries == 1:
                print(f'Send the model compilation results to jenkins：')
            else:
                print(
                    f'Try sending the model compilation results to jenkins a {retries}th time:')
            break
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

            retries += 1
            time.sleep(1)

    queue_id = response.headers['Location'].split('/')[-2]
    job_url = get_job_url(queue_id, auth)
    print(job_url)
    assert job_url


def get_json_status_params(file, toolchain, cmt_id, target, mode):
    if len(file) == 0:
        logging.info(f'No {file}!')
        return []

    with open(file[0], 'r') as f:
        data = json.load(f)
    f.close()

    if len(data) == 0 or len(data["failed_cases"]) == 0:
        return []
    else:
        failed_case_names = [case for case in data["failed_cases"]]

    failed_cases = case_name_to_path(failed_case_names)

    type_dict = {0: "build", 1: "efficiency_build", 2: "accuracy_build"}
    nntc_type = type_dict.get(mode)

    failed_params = [
        {"case_name": case_name, "toolchain": toolchain, "commit_sha": cmt_id, "target": target,
         "mode": nntc_type, "status": 1} for case_name in set(failed_cases)]

    return failed_params

def read_json_and_post_to_jenkins(target, sdk, git_job_id):
    toolchain = sdk.split('_')[0]
    cmt_id = sdk.split('_')[1]

    if toolchain == 'tpu-mlir':
        mlir_file = glob.glob(f'mlir_{target}_acc_cases_status.json')
        mlir_params = get_json_status_params(mlir_file, toolchain, cmt_id, target, mode=0)
        status_params = mlir_params
    elif toolchain == 'nntoolchain':
        nntc_eff_file = glob.glob(f'nntc_{target}_eff_cases_status.json')
        nntc_acc_file = glob.glob(f'nntc_{target}_acc_cases_status.json')
        nntc_eff_params = get_json_status_params(nntc_eff_file, toolchain, cmt_id, target, mode=1)
        nntc_acc_params = get_json_status_params(nntc_acc_file, toolchain, cmt_id, target, mode=2)
        status_params = nntc_eff_params + nntc_acc_params
    else:
        logging.error('Invalid Toolchain')
        sys.exit(-1)

    if len(status_params) == 0:
        logging.error('No model compilation failed on {target}')
        return

    github_url = f'https://github.com/sophgo/model-zoo/actions/runs/{git_job_id}'
    case_params = {"cases": status_params, "github_url": github_url}
    params = f"{case_params}"

    json_param = {'JSON_PARAM': params}
    post_jenkins(json_param)

    json_str = json.dumps(case_params, indent=4)
    print(json_str)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Post failed cases to blame regression test')
    parser.add_argument(
        '--sdk', type=str, help='the version of toolchain sdk')
    parser.add_argument(
        '--git_job_id', type=str, help='the job id of github action')
    args = parser.parse_args()

    for target in ['BM1684', 'BM1684X']:
        read_json_and_post_to_jenkins(target, args.sdk, args.git_job_id)

if __name__ == '__main__':
    main()
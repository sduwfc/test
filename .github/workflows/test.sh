#!/bin/bash

set -eE
trap 'echo "ERROR: $BASH_SOURCE:$LINENO $BASH_COMMAND" >&2' ERR

function quiet_exec()
{
    log=/tmp/log.txt
    echo $@
    $@ 2>&1 | cat > $log
    test ${PIPESTATUS[0]} -eq 0 || {
        cat $log
        echo task failed
        echo $@
        return 1
    }
}

cmd="python3 -m pytest -s"
while getopts "m:t:-:" arg; do
  case $arg in
    -)
      case ${OPTARG} in
        full)
          cases="--full"
          ;;
        *)
          echo "Unsupported option --${OPTARG}"
          exit -1
          ;;
      esac
      ;;
    t)
      cmd="${cmd} --target=${OPTARG}"
      ;;
    m)
      cmd="${cmd} -m \"${OPTARG}\""
      ;;
  esac
done

[ -z "$cases" ] && cases=${@:$OPTIND}
if [[ ! -z $cases ]]; then
    echo "Testing \"$cases\""
    export TEST_CASES=$cases
fi

quiet_exec pip3 install -r test/requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple
eval ${cmd} test

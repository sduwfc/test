#!/bin/bash

DIR="$(dirname "$(realpath "$0")")"
DIR="$(realpath $DIR/../..)"

fn_list=/tmp/lfs-files.list

function check()
{
    local fn=$1
    grep ${fn#./} $fn_list > /dev/null || {
        echo $fn should upload via lfs, not found in ls-files
        exit 1
    }
    git check-attr --all -- $fn  | grep lfs > /dev/null || {
        echo $fn should upload via lfs
        exit 1
    }
}

git lfs ls-files > $fn_list

while read fn; do check $fn; done < <(find -type f -name *.caffemodel)
while read fn; do check $fn; done < <(find -type f -name *.prototxt)
while read fn; do check $fn; done < <(find -type f -name *.pt)
while read fn; do check $fn; done < <(find -type f -name *.onnx)
while read fn; do check $fn; done < <(find -type f -name *.pb)
while read fn; do check $fn; done < <(find -type f -name *.tflite)
while read fn; do check $fn; done < <(find -type f -name *.JPG)
while read fn; do check $fn; done < <(find -type f -name *.pdmodel)
while read fn; do check $fn; done < <(find -type f -name *.pdiparams)
while read fn; do check $fn; done < <(find -type f -name *.zip)
while read fn; do check $fn; done < <(find -type f -name *.mdb)
while read fn; do check $fn; done < <(find -type f -name *.png)
while read fn; do check $fn; done < <(find -type f -name *.tar)
while read fn; do check $fn; done < <(find dataset -type f -name *.jpg)
while read fn; do check $fn; done < <(find $DIR -type f -size +1M ! -path *.git*)

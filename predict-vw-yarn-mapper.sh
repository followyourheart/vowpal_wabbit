#!/bin/bash

set -e

pwd=$(cd $(dirname $0); pwd)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pwd}/libs/

MODEL_NAME="${1}"

CMD_PREDICT="
	./vw 
	-t 
	-i ${MODEL_NAME} 
	-d /dev/stdin 
	-p /dev/stdout
"

${CMD_PREDICT}
#!/bin/bash

set -e

pwd=$(cd $(dirname $0); pwd)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pwd}/libs/

TEST_RUN="${1}"

OPTIMIZATION_TYPE="${2}"

if [[ "${OPTIMIZATION_TYPE}" == "sgd" ]]; then
	BIT_PRECISION_SGD="${3}"
	PASSES_SGD="${4}"
	LOSS_FUNC_SGD="${5}"
	REGULARIZATION_TYPE_SGD="${6}"
	REGULARIZATION_VALUE_SGD="${7}"

	if [[ "${REGULARIZATION_TYPE_SGD}" == "" || "${REGULARIZATION_VALUE_SGD}" == "" ]]; then
		unset REGULARIZATION_VALUE_SGD
		REGULARIZATION_SGD=""
	elif [[ "${REGULARIZATION_TYPE_SGD}" == "l1" ]]; then
		REGULARIZATION_SGD="${REGULARIZATION_VALUE_SGD:+--l1 ${REGULARIZATION_VALUE_SGD}}"
	elif [[ "${REGULARIZATION_TYPE_SGD}" == "l2" ]]; then
		REGULARIZATION_SGD="${REGULARIZATION_VALUE_SGD:+--l2 ${REGULARIZATION_VALUE_SGD}}"
	fi
	if [[ "${REGULARIZATION_SGD}" == "" ]]; then
		unset REGULARIZATION_SGD
	fi
elif [[ "${OPTIMIZATION_TYPE}" == "bfgs" ]]; then
	BIT_PRECISION_BFGS="${3}"
	PASSES_BFGS="${4}"
	LOSS_FUNC_BFGS="${5}"
	REGULARIZATION_TYPE_BFGS="${6}"
	REGULARIZATION_VALUE_BFGS="${7}"

	if [[ "${REGULARIZATION_TYPE_BFGS}" == "" || "${REGULARIZATION_VALUE_BFGS}" == "" ]]; then
		unset REGULARIZATION_VALUE_BFGS
		REGULARIZATION_BFGS=""
	elif [[ "${REGULARIZATION_TYPE_BFGS}" == "l1" ]]; then
		REGULARIZATION_BFGS="${REGULARIZATION_VALUE_BFGS:+--l1 ${REGULARIZATION_VALUE_BFGS}}"
	elif [[ "${REGULARIZATION_TYPE_BFGS}" == "l2" ]]; then
		REGULARIZATION_BFGS="${REGULARIZATION_VALUE_BFGS:+--l2 ${REGULARIZATION_VALUE_BFGS}}"
	fi
	if [[ "${REGULARIZATION_BFGS}" == "" ]]; then
		unset REGULARIZATION_BFGS
	fi
else
	BIT_PRECISION_SGD="${3}"
	PASSES_SGD="${4}"
	LOSS_FUNC_SGD="${5}"
	REGULARIZATION_TYPE_SGD="${6}"
	REGULARIZATION_VALUE_SGD="${7}"

	BIT_PRECISION_BFGS="${8}"
	PASSES_BFGS="${9}"
	LOSS_FUNC_BFGS="${10}"
	REGULARIZATION_TYPE_BFGS="${11}"
	REGULARIZATION_VALUE_BFGS="${12}"

	if [[ "${REGULARIZATION_TYPE_SGD}" == "" || "${REGULARIZATION_VALUE_SGD}" == "" ]]; then
		unset REGULARIZATION_VALUE_SGD
		REGULARIZATION_SGD=""
	elif [[ "${REGULARIZATION_TYPE_SGD}" == "l1" ]]; then
		REGULARIZATION_SGD="${REGULARIZATION_VALUE_SGD:+--l1 ${REGULARIZATION_VALUE_SGD}}"
	elif [[ "${REGULARIZATION_TYPE_SGD}" == "l2" ]]; then
		REGULARIZATION_SGD="${REGULARIZATION_VALUE_SGD:+--l2 ${REGULARIZATION_VALUE_SGD}}"
	fi
	if [[ "${REGULARIZATION_SGD}" == "" ]]; then
		unset REGULARIZATION_SGD
	fi

	if [[ "${REGULARIZATION_TYPE_BFGS}" == "" || "${REGULARIZATION_VALUE_BFGS}" == "" ]]; then
		unset REGULARIZATION_VALUE_BFGS
		REGULARIZATION_BFGS=""
	elif [[ "${REGULARIZATION_TYPE_BFGS}" == "l1" ]]; then
		REGULARIZATION_BFGS="${REGULARIZATION_VALUE_BFGS:+--l1 ${REGULARIZATION_VALUE_BFGS}}"
	elif [[ "${REGULARIZATION_TYPE_BFGS}" == "l2" ]]; then
		REGULARIZATION_BFGS="${REGULARIZATION_VALUE_BFGS:+--l2 ${REGULARIZATION_VALUE_BFGS}}"
	fi
	if [[ "${REGULARIZATION_BFGS}" == "" ]]; then
		unset REGULARIZATION_BFGS
	fi
fi

####
# MR1 sets $mapred_map_tasks
# MR2/YARN sets $mapreduce_job_maps
nmappers=${mapreduce_job_maps}

# MR1 sets $mapreduce_job_submithost
# MR2/YARN sets $mapreduce_job_submithostname
submit_host=${mapreduce_job_submithostname}

# MR1 sets $mapred_output_dir
# MR2/YARN sets $mapreduce_output_fileoutputformat_outputdir
output_dir=${mapreduce_output_fileoutputformat_outputdir}

set -u

# This works on both MR1 and MR2/YARN
mapper=`printenv mapred_task_id | cut -d "_" -f 5`
mapred_job_id=`echo "${mapred_job_id}" | awk -F "_" '{print $NF}'`

# debug
echo "mapred_task_id: ${mapper}" > /dev/stderr
echo "#mappers: ${nmappers}" > /dev/stderr
echo "out dir: ${output_dir}" > /dev/stderr
echo "host: ${submit_host}" > /dev/stderr

TEMP_CACHE="vw.tmp.cache"
rm -f ${TEMP_CACHE} || true

echo 'Starting training' > /dev/stderr

if [[ "${OPTIMIZATION_TYPE}" == "sgd" ]]; then
	echo "Train model only using SGD..." > /dev/stderr

	CMD_SGD="
		./vw 
		--total ${nmappers} 
		--node ${mapper} 
		--unique_id ${mapred_job_id} 
		--span_server ${submit_host} 
		--save_per_pass 
		--noconstant 
		--holdout_off 
		-b ${BIT_PRECISION_SGD:-20} 
		--passes ${PASSES_SGD:-10} 
		--termination ${TERMINATION:-0.00001} 
		--cache_file ${TEMP_CACHE} 
		-d /dev/stdin 
		-f sgd.vw.model 
		--readable_model sgd.vw.rmodel 
		${LOSS_FUNC_SGD:+--loss_function=${LOSS_FUNC_SGD}} 
		${REGULARIZATION_SGD:+${REGULARIZATION_SGD}}
	"
elif [[ "${OPTIMIZATION_TYPE}" == "bfgs" ]]; then
	echo "Train model only using BFGS..." > /dev/stderr
	CMD_BFGS="
		./vw 
		--total ${nmappers} 
		--node ${mapper} 
		--unique_id ${mapred_job_id} 
		--span_server ${submit_host} 
		--save_per_pass 
		--noconstant 
		--bfgs 
		--mem 5 
		--holdout_off 
		-b ${BIT_PRECISION_BFGS:-20} 
		--passes ${PASSES_BFGS} 
		--termination ${TERMINATION:-0.00001} 
		--cache_file ${TEMP_CACHE} 
		-d /dev/stdin 
		-f bfgs.vw.model 
		--readable_model bfgs.vw.rmodel 
		${LOSS_FUNC_BFGS:+--loss_function=${LOSS_FUNC_BFGS}} 
		${REGULARIZATION_BFGS:+${REGULARIZATION_BFGS}}
	"
else
	echo "Train model with two step, first SGD, second BFGS..." > /dev/stderr
	echo "Step 1. SGD..." > /dev/stderr
	CMD_SGD="
		./vw 
		--total ${nmappers} 
		--node ${mapper} 
		--unique_id ${mapred_job_id} 
		--span_server ${submit_host} 
		--save_per_pass 
		--noconstant 
		--holdout_off 
		-b ${BIT_PRECISION_SGD:-20} 
		--passes ${PASSES_SGD:-10} 
		--termination ${TERMINATION:-0.00001} 
		--cache_file ${TEMP_CACHE} 
		-d /dev/stdin 
		-f sgd.vw.model 
		--readable_model sgd.vw.rmodel 
		${LOSS_FUNC_SGD:+--loss_function=${LOSS_FUNC_SGD}} 
		${REGULARIZATION_SGD:+${REGULARIZATION_SGD}}
	"

	echo "Step 2. BFGS..." > /dev/stderr
	mapred_job_id=`expr $mapred_job_id \* 2` #create new nonce
	CMD_BFGS="
		./vw 
		--total ${nmappers} 
		--node ${mapper} 
		--unique_id ${mapred_job_id} 
		--span_server ${submit_host} 
		--save_per_pass 
		--bfgs 
		--mem 5 
		--noconstant 
		--holdout_off 
		-b ${BIT_PRECISION_BFGS:-20} 
		--passes ${PASSES_BFGS:-10} 
		--termination ${TERMINATION:-0.00001} 
		--cache_file ${TEMP_CACHE} 
		-i sgd.vw.model 
		-f bfgs.vw.model 
		--readable_model bfgs.vw.rmodel 
		${LOSS_FUNC_BFGS:+--loss_function=${LOSS_FUNC_BFGS}} 
		${REGULARIZATION_BFGS:+${REGULARIZATION_BFGS}}
	"
fi

#
train_vw_log="vw.train.log"
if [[ "${mapper}" == "000000" ]]; then
	if [[ "${TEST_RUN}" == "n" ]]; then
		if [[ "${OPTIMIZATION_TYPE}" == "sgd" ]]; then
			echo "master '${mapper}' run SGD..." > /dev/stderr
			${CMD_SGD} 1>/dev/stdout 2>/dev/stderr
		elif [[ "${OPTIMIZATION_TYPE}" == "bfgs" ]]; then
			echo "master '${mapper}' run BFGS ..." > /dev/stderr
			${CMD_BFGS} 1>/dev/stdout 2>/dev/stderr
		else
			echo "master '${mapper}' run SGD..." > /dev/stderr
			${CMD_SGD} 1>/dev/stdout 2>/dev/stderr

			echo "master '${mapper}' run BFGS ..." > /dev/stderr
			${CMD_BFGS} 1>/dev/stdout 2>/dev/stderr
		fi
	else
		echo "Test..." > /dev/stderr
		echo "master '${mapper}' run SGD:${CMD_SGD}" > /dev/stderr
		echo "master '${mapper}' run BFGS:${CMD_BFGS}" > /dev/stderr
		cat > /dev/null
	fi

	if [[ $? -ne 0 ]]; then
		exit 5
	fi

	# store models and output into hdfs
	hadoop fs -put -f *.vw.*model ${output_dir} || true
	hadoop fs -put -f vw.* ${output_dir} || true
else 
	if [[ "${TEST_RUN}" == "n" ]]; then
		if [[ "${OPTIMIZATION_TYPE}" == "sgd" ]]; then
			echo "worker '${mapper}' run SGD..." > /dev/stderr
			${CMD_SGD} 1>/dev/stdout 2>/dev/stderr
		elif [[ "${OPTIMIZATION_TYPE}" == "bfgs" ]]; then
			echo "worker '${mapper}' run BFGS ..." > /dev/stderr
			${CMD_BFGS} 1>/dev/stdout 2>/dev/stderr
		else
			echo "worker '${mapper}' run SGD..." > /dev/stderr
			${CMD_SGD} 1>/dev/stdout 2>/dev/stderr

			echo "worker '${mapper}' run BFGS ..." > /dev/stderr
			${CMD_BFGS} 1>/dev/stdout 2>/dev/stderr
		fi
	else
		echo "Test..." > /dev/stderr
		echo "worker '${mapper}' run SGD:${CMD_SGD}" > /dev/stderr
		echo "worker '${mapper}' run BFGS:${CMD_BFGS}" > /dev/stderr
		cat > /dev/null
	fi

	if [[ $? -ne 0 ]]; then
		exit 6
	fi
fi
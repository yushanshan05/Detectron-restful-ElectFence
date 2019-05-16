#coding=utf-8
#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on one or more datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2	# NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import pprint
import sys
import time
import datetime

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from core.config import merge_cfg_from_list
from core.test_engine import run_inference
import utils.c2
import utils.logging

utils.c2.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def parse_args():
	parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
	parser.add_argument(
		'--cfg',
		dest='cfg_file',
		help='optional config file',
		default=None,
		type=str
	)
	parser.add_argument(
		'--wait',
		dest='wait',
		help='wait until net file exists',
		default=True,
		type=bool
	)
	parser.add_argument(
		'--vis', dest='vis', help='visualize detections', action='store_true'
	)
	parser.add_argument(
		'--multi-gpu-testing',
		dest='multi_gpu_testing',
		help='using cfg.NUM_GPUS for inference',
		action='store_true'
	)
	parser.add_argument(
		'--range',
		dest='range',
		help='start (inclusive) and end (exclusive) indices',
		default=None,
		type=int,
		nargs=2
	)
	parser.add_argument(
		'opts',
		help='See lib/core/config.py for all options',
		default=None,
		nargs=argparse.REMAINDER
	)

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	return parser.parse_args()

'''
test_result: 根据测试结果打印的信息，统计每个类别的AP、耗时、AP、AP50、AP75、APs、APm、APl
input：
	modelName：模型名称，如model_iter24999.pkl
	logPath：测试结果写入的文件名称，比如"log/oil/test.txt"
	dataType：字符串，表示该数据是训练集还是验证集(train 或 val)
'''
def test_result(modelName, logPath, dataType):
	today = datetime.date.today()  
	#print (" 111111111 ", type( str(dataType)))
	testResult_Path = str( today ) + "_" + str(dataType) + ".txt"
	#print testResult_Path

	fw = open(testResult_Path, 'a')	
	totalTime = 0
	classAPLst = []
	APLst = []
	#需保存的数据是：模型、每个类别的AP、耗时、AP、AP50、AP75、APs、APm、APl
	with open(logPath, 'r') as ann_file:
		lines = ann_file.readlines()
		for l in lines:
			#total time
			if l.startswith("INFO test_engine.py: 162: Total inference time:"):
				lst = l.strip().split(":")
				totalTime = lst[-1]
				
			#每个类别的AP
			if l.startswith("INFO json_dataset_evaluator.py: 231:"):
				lst = l.strip().split(":")
				ap = lst[-1]
				classAPLst.append( ap ) #'\t' tab
				
			#AP、AP50、AP75、APs、APm、APl
			if l.startswith("INFO task_evaluation.py: 186: copypaste:"):
				lst = l.strip().split(":")
				ll = lst[-1]
				APLst = ll.strip().split(",")
	#fw.write(str(len(classAPLst)) + "\n")	
	
	#print ( totalTime )		
	print ( classAPLst )
	#print ( APLst )	
	fw.write(modelName + "\n")	
	fw.write(str(dataType) + " " + str(datetime.datetime.now()) + "\n")
	fw.write(totalTime + "\n")
	#fw.write("AP、AP50、AP75、APs、APm、APl:\n")
	fw.write("	".join(APLst) + "\n")
	fw.write("class AP:\n")
	for l in classAPLst:	
		fw.write(l + "\n")
		
	classAPLst = []
	
	fw.write("\n\n")
	fw.close()

	
#test_result("model_iter9999.pkl", "test.txt", "val")

	
if __name__ == '__main__':
	workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
	logger = utils.logging.setup_logging("testzj.log")
	args = parse_args()
	logger.info('Called with args:')
	logger.info(args)
	if args.cfg_file is not None:
		merge_cfg_from_file(args.cfg_file)
	if args.opts is not None:
		merge_cfg_from_list(args.opts)
	assert_and_infer_cfg()
	logger.info('Testing with config:')
	logger.info(pprint.pformat(cfg))

	while not os.path.exists(cfg.TEST.WEIGHTS) and args.wait:
		logger.info('Waiting for \'{}\' to exist...'.format(cfg.TEST.WEIGHTS))
		time.sleep(10)
	
	l = cfg.TEST.WEIGHTS
	lst = l.strip().split("/")
	modelPath = "/".join(lst[:len(lst)-1]) + "/"

	print ( modelPath )
	files = [x for x in os.listdir(modelPath) if os.path.isfile(modelPath+x) and x.endswith('.pkl')]
	modelNumLst = []
	for l in files:
		ll = l[10:len(l)-4]
		#print ( ll )
		modelNumLst.append( int(ll) )
	
	modelNumLst.sort()
	for ll in modelNumLst:
		l = "model_iter" + str(ll) + ".pkl"
	
		run_inference(
			modelPath+l,
			ind_range=args.range,
			multi_gpu_testing=args.multi_gpu_testing,
			check_expected_results=True,
			)
		test_result(l, "testzj.log", cfg.TEST.DATASETS)
	#os.remove( "testzj.log" )
		
		

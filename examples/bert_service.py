#!/usr/bin/env python
# visit https://tool.lu/pyc/ for more information
from PIL import Image
import numpy as np
from model_service.tfserving_model_service import TfServingBaseService
from tensorflow import make_tensor_proto
from tensorflow_serving.apis import predict_pb2
from multiprocessing import Lock
import random
from tensorflow.python.tools import saved_model_cli
import tensorflow as tf


class mnist_service(TfServingBaseService):
	mutex = Lock()
	buffer = []
	
	def __init__(self, model_name, model_path):
		self.model_name = model_name
		self.model_path = model_path
		signature_def_map = saved_model_cli.get_signature_def_map(model_path, tf.saved_model.tag_constants.SERVING)
		self.model = Model_Sig()
		signature = []
		for name in signature_def_map:
			signature.append(name)
		
		if len(signature) == 1:
			self.model.model_signature = signature[0]
		else:
			self.model.model_signature = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
		self.model.model_name = self.model_name
		self.stub = get_tf_server_stub()
	
	def _preprocess(self, data):
		print(data)
		preprocessed_data = {}
		for k, v in data.items():
			for file_name, file_content in v.items():
				image1 = Image.open(file_content)
				image1 = np.array(image1, dtype=np.float32)
				image1.resize((1, 784))
				preprocessed_data[k] = image1
				with mnist_service.mutex:
					mnist_service.buffer.append(image1[0])
		
		return preprocessed_data
	
	def _inference(self, data):
		result = {}
		with mnist_service.mutex:
			data = np.array(mnist_service.buffer, dtype=np.float32)
			mnist_service.buffer = []
		if len(data) > 0:
			request = predict_pb2.PredictRequest()
			request.model_spec.name = self.model.model_name
			request.model_spec.signature_name = self.model.model_signature
			print(data.shape)
			request.inputs['images'].CopyFrom(make_tensor_proto(data))
			response = self.stub.Predict(request, 60)
			for output_name in response.outputs:
				tensor_proto = response.outputs[output_name]
				result[output_name] = tf.contrib.util.make_ndarray(tensor_proto).tolist()
			
			return result
		return None
	
	def _postprocess(self, data):
		return data


class Model_Sig:
	
	def __init__(self):
		self.model_signature = ''
		self.model_name = ''

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import tensorflow as tf
import numpy as np
import json
import os


GRPC_PORT = "8500"
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

def serve_grpc():
    channel = grpc.insecure_channel(f'http://ranking-f259dd61b9-predictor-default.search/v1/models/d2:predict:{GRPC_PORT}')
    stub = prediction_service_pb2.PredictionServiceStub(channel)
    grpc_request = predict_pb2.PredictRequest()
    # grpc_request.model_spec.signature_name = 'recommendation'
    f = open('sample_tf_scann_input.json')
    data = json.load(f)
    tags_vector_np = np.array(data['inputs']['tags_vector'])
    type_vector_np = np.array(data['inputs']['type_vector'])
    field_weight_np = np.array(data['inputs']['field_weights'])
    grpc_request.inputs['tags_vector'].CopyFrom(tf.make_tensor_proto(tags_vector_np.tolist(), shape=tags_vector_np.shape))
    grpc_request.inputs['type_vector'].CopyFrom(tf.make_tensor_proto(type_vector_np.tolist(), shape=type_vector_np.shape))
    grpc_request.inputs['field_weights'].CopyFrom(tf.make_tensor_proto(field_weight_np.tolist(), shape=field_weight_np.shape))
    
    predictions = stub.Predict(grpc_request, 10)

    print("predictionssss", predictions)
    outputs_tensor_proto = predictions.outputs["output_1"]
    shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
    outputs = np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())
    print("hello", outputs)

serve_grpc()
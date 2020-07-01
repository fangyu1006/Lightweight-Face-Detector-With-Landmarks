from  onnx_tf.backend import prepare
import onnx

model_onnx = onnx.load('./converted_models/mobilenet/mobilenet_sim.onnx')

# prepare model for exporting to tensorflow using tensorflow backend
tf_rep = prepare(model_onnx)
#print(tf_rep.run(dummy_input)) # run sample inference of model
print(tf_rep.inputs) # input nodes to the model
print('------')
print(tf_rep.outputs) # output nodes from the model
print('-----')
print(tf_rep.tensor_dict) # all nodes in the model

# export tensorflow backend to tensorflow tf file
tf_rep.export_graph('./converted_models/mobilenet/mobilenet.pb')

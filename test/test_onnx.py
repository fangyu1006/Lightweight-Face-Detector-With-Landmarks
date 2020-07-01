import onnxruntime as nxrun
import numpy as np

ximg = np.random.rand(1,3,240,320).astype(np.float32)
sess = nxrun.InferenceSession("./converted_models/mobilenet/mobilenet_sim.onnx")

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
result = sess.run(None, {input_name: ximg})
print(result)

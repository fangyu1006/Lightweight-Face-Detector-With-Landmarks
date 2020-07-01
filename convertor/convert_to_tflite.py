import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_frozen_graph('./converted_models/mobilenet/mobilenetv3.pb',
                                                      input_arrays=['input0'],
                                                      output_arrays=['Concat_223', 'Concat_198', 'Concat_248'],
                                                      input_shapes={"input0":[1,240,320,3]})

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.allow_custom_ops=True
converter.inference_type = tf.float32
converter.inference_input_type = tf.float32
tf_lite_model = converter.convert()
open('./converted_models/mobilenet/mobilenetv3.tflite', 'wb').write(tf_lite_model)

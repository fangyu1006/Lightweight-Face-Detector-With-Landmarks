import tensorflow as tf

def freeze_graph(input_checkpoint, output_node_names, file_save):
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            path_meta = input_checkpoint+'.meta'
            saver = tf.compat.v1.train.import_meta_graph(path_meta)
            saver.restore(sess, input_checkpoint)

            output_graph_def = tf.graph_util.convert_variables_to_constants(
                    sess,
                    tf.compat.v1.get_default_graph().as_graph_def(),
                    output_node_names.split(",")
                    )
            with tf.io.gfile.GFile(file_save, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            for v in output_graph_def.node:
                print(v.name)
            print("save pb file to " + file_save)
            

freeze_graph("/home/fangyu/git/Face-Detector-1MB-with-landmark/converted_models/mobilenet/mobilenetv3.ckpt", "input0,Concat_198,Concat_223,Concat_248","/home/fangyu/git/Face-Detector-1MB-with-landmark/converted_models/mobilenet/mobilenetv3.pb")

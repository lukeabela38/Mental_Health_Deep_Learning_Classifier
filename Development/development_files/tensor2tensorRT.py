
import os
import tensorflow as tf

class Tensor2TensorRT:
    def __init__(self, input_model_dir = "Models"):

        self.input_model_dir = input_model_dir

        self.params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP16')

    def _optimise(self, output_model_dir = "ModelsRT/"):
        if os.path.exists(output_model_dir) != True:
            os.makedirs(output_model_dir)
        
        converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=self.input_model_dir, conversion_params=self.params)
        converter.convert()
        converter.save(output_model_dir)

if __name__ == "__main__":
    t2t = Tensor2TensorRT()
    t2t._optimise()
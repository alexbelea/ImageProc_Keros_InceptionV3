import tensorflow as tf

# Check if TensorFlow can detect a GPU with CUDA and cuDNN
print("TensorFlow version:", tf.__version__)
print("Is GPU available:", tf.config.list_physical_devices('GPU'))
print("GPU device name:", tf.test.gpu_device_name())

# Check if cuDNN is installed and being used
print("Is built with CUDA:", tf.test.is_built_with_cuda())
try:
    print("Is cuDNN available:", tf.test.is_built_with_cudnn())
    print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])
except:
    print("Unable to determine cuDNN status/version")
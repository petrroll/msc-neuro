import tensorflow as tf

if __name__ == "__main__":
    print(tf.VERSION)
    print("GPU:", tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
    ))


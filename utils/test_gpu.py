import tensorflow as tf


def run():
    '''
    Tests whether tensorflow sees available GPU.
    '''

    print(tf.VERSION)
    print("GPU:", tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
    ))


if __name__ == "__main__":
    run()

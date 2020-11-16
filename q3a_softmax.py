import numpy as np
from scipy.special import softmax as ground_truth_softmax #used for tests only

def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    # reminder: softmax(x) = softmax(x + c)
    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        # Rescale rows
        x = x - np.transpose(np.expand_dims(np.max(x, axis=1), 0))
        # Calc softmax along rows
        exp_sum = np.sum(np.exp(x), axis=1)
        x = np.exp(x) / np.expand_dims(exp_sum, 1)
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        x = x - np.max(x)
        exp_sum = np.sum(np.exp(x))
        x = np.exp(x)/exp_sum
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


def test_your_softmax_test():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    ### YOUR OPTIONAL CODE HERE
    x = np.array([[1, 0.5, 0.2, 3],
                  [1, -1, 7, 3],
                  [2, 12, 13, 3]])
    test4 = softmax(x)
    print(test4)
    ans4 = np.array([[1.05877e-01, 6.42177e-02, 4.75736e-02, 7.82332e-01],
                     [2.42746e-03, 3.28521e-04, 9.79307e-01, 1.79366e-02],
                     [1.22094e-05, 2.68929e-01, 7.31025e-01, 3.31885e-05]])
    assert np.allclose(test4, ans4, rtol=1e-05, atol=1e-06)

    for _ in range(500):
        h = np.random.randint(1, 100)
        w = np.random.randint(1, 100)
        x = np.random.rand(h, w)
        test5 = softmax(x)
        ans5 = ground_truth_softmax(x, axis=1)
        assert np.allclose(test5, ans5, rtol=1e-05, atol=1e-06)
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    test_your_softmax_test()

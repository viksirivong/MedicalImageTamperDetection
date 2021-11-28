import tensorflow as tf
import numpy as np

DATA_LOAD_PATH  = None
MODEL_NAME      = 'tamperDetection.h5'

def main():
    # load data.
    with np.load(DATA_LOAD_PATH + '\\devtest.npz') as data:
        # partition the desired amount from the test set.
        test_X = data['devtest_X']
        test_Y = data['devtest_Y']

    # load model.
    model = tf.keras.models.load_model(MODEL_NAME)

    # make predictions on test set.
    predictions = model.predict(test_X)
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5]  = 0

    # print predictions on test set along with ground truth labels.
    nMisclassifications = 0
    for i in range(len(predictions)):
        prediction   = 'tampered' if predictions[i] == 1        else 'not tampered'
        ground_truth = 'tampered' if test_Y[i] == 1             else 'not tampered'
        result       = 'correct'  if prediction == ground_truth else 'wrong'
        print('[{}] prediction={}, truth={}, {}'.format(i, prediction, ground_truth, result))
        if prediction != ground_truth:
            nMisclassifications += 1

    # calculate and print accuracy on test set.
    accuracy = (len(predictions) - nMisclassifications) / len(predictions)
    print("\nmodel's accuracy on test set: {:.4f}".format(accuracy))

if __name__ == "__main__":
    main()
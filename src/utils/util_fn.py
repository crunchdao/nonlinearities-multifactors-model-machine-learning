import numpy as np
import sklearn.metrics


def create_y_hat_new(predictions, b_matrix):
    """Extracts predictions of a particular moon
    Args:
        predictions:  shape : f_matrix.shape[0],target_shape:1

    Returns: shape : f_matrix.date.unique(),
    """
    predictions_parallel = np.dot(
        b_matrix.to_numpy(), np.dot(np.linalg.pinv(b_matrix), predictions)
    )
    predictions -= predictions_parallel
    y_hat_new = predictions
    return y_hat_new


def fitness(predictions, b_matrix, targets):
    """Is executed after each iteration
    predictions: f_matrix.shape[0],



    """
    y_hat_new = create_y_hat_new(predictions, b_matrix)
    loss = sklearn.metrics.mean_squared_error(y_hat_new, targets)
    return loss


def fitness_moon(predictions, b_matrix, targets):
    """ """
    epochs = predictions.date.unique()
    loss = 0
    for epoch in epochs:
        b_matrix_moon_wise = b_matrix[b_matrix["date"] == epoch]
        target_moon_wise = targets[targets["date"] == epoch]
        prediction_moon_wise = predictions[predictions["date"] == epoch]
        loss += fitness(prediction_moon_wise, b_matrix_moon_wise, target_moon_wise)

    return loss

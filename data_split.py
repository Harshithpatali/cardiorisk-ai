"""Create the untouched stratified test partition."""
import numpy as np
from sklearn.model_selection import train_test_split


def split_holdout(features, target, test_size, random_seed):
    indices = np.arange(len(features))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_seed,
        stratify=target,
    )
    x_train = features.iloc[train_indices].reset_index(drop=True)
    x_test = features.iloc[test_indices].reset_index(drop=True)
    y_train = np.asarray(target)[train_indices]
    y_test = np.asarray(target)[test_indices]
    return x_train, x_test, y_train, y_test

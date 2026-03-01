import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess(df, scaler: MinMaxScaler, kind: str):
    """Return normalized data."""
    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError("Data must be a 2-D array")

    if np.any(sum(np.isnan(df)) != 0):
        print("Data contains null values. Will be replaced with 0")
        df = np.nan_to_num()

    if kind == "train":
        df = scaler.fit_transform(df)
    elif kind == "test":
        df = scaler.transform(df)

    print("Data normalized")
    return df


def minibatch_slices_iterator(length, step_size, ignore_incomplete_batch=False):
    """Iterate through all mini-batch slices."""
    start = 0
    stop1 = (length // step_size) * step_size
    while start < stop1:
        yield slice(start, start + step_size, 1)
        start += step_size
    if not ignore_incomplete_batch and start < length:
        yield slice(start, length, 1)


class BatchSlidingWindow(object):
    """Class for obtaining mini-batch iterators of sliding windows."""

    def __init__(
        self,
        array_size,
        window_size,
        step_size,
        batch_size,
        excludes=None,
        shuffle=False,
        ignore_incomplete_batch=False,
    ):
        if window_size < 1:
            raise ValueError("`window_size` must be at least 1")
        if array_size < window_size:
            raise ValueError("`array_size` must be at least as large as `window_size`")
        if excludes is not None:
            excludes = np.asarray(excludes, dtype=np.bool)
            expected_shape = (array_size,)
            if excludes.shape != expected_shape:
                raise ValueError(
                    "The shape of `excludes` is expected to be {}, but got {}".format(
                        expected_shape, excludes.shape
                    )
                )

        if excludes is not None:
            mask = np.logical_not(excludes)
        else:
            mask = np.ones([array_size], dtype=np.bool)
        mask[: window_size - 1] = False
        where_excludes = np.where(excludes)[0]
        for k in range(1, window_size):
            also_excludes = where_excludes + k
            also_excludes = also_excludes[also_excludes < array_size]
            mask[also_excludes] = False

        indices = np.arange(array_size)[mask]
        self._indices = indices.reshape([-1, 1])
        self._offsets = np.arange(-window_size + 1, 1)

        self._array_size = array_size
        self._window_size = window_size
        self._step_size = step_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._ignore_incomplete_batch = ignore_incomplete_batch

    def get_iterator(self, arrays):
        """Iterate through sliding windows of each array in `arrays`."""
        arrays = tuple(np.asarray(a) for a in arrays)
        if not arrays:
            raise ValueError("`arrays` must not be empty")

        if self._shuffle:
            np.random.shuffle(self._indices)

        for s in minibatch_slices_iterator(
            length=len(self._indices),
            step_size=self._step_size,
            ignore_incomplete_batch=self._ignore_incomplete_batch,
        ):
            idx = self._indices[s] + self._offsets
            yield tuple(a[idx] if len(a.shape) == 1 else a[idx, :] for a in arrays)

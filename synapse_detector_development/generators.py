import numpy as np

def augmenting_generator(raw_data, gt_data, dists, input_shape, output_shape,
                         batch_size, ignore_distance=35.0, channel_idx=0):
    """Create a generator that augments data for training.

    This generator chooses random subsets of the supplied data and has a chance
    to randomly flip the images along one or more axes or scale/shift the
    grayscale intensities.

    Parameters
    ----------
    raw_data : array-like
        Volume containing the raw images.
    gt_data : array-like
        Volume containing the ground truth.
    dists : array-like
        Volume containing the element-wise distance from each 0-element of
        ``gt`` to the nearest non-zero element.
    input_shape : tuple of int
        The shape of the data to feed into the neural network model.
    output_shape : tuple of int
        The expected shape of the neural network model output.
    batch_size : int
        The number of augmented images to include in a batch.
    ignore_distance : float, default 35.0
        The maximum average distance that augmented data is allowed to be from
        the ground truth. This prevents training on a subset of the data that
        contains no ground truth.
    channel_idx : int, default 0
        The index of the image channel dimension, typically first or last
        depending on the neural network framework.

    Yields
    ------
    aug : array-like
        The ``batch_size``-by-``input_shape`` augmented data to feed into a
        neural network.
    gt : array-like
        The ``output_shape`` ground truth subvolume associated with ``aug``.
    """
    # drop the channel idx
    input_shape = [d for i, d in enumerate(input_shape) if i != channel_idx]
    output_shape = [d for i, d in enumerate(output_shape) if i != channel_idx]
    raw_to_gt_offsets = [(i - o) // 2
                         for i, o in zip(input_shape, output_shape)]

    while True:
        batch = []
        for idx in range(batch_size):
            while True:
                lo_corner = [np.random.randint(rd - i)
                             for rd, i in zip(raw_data.shape, input_shape)]
                slices = [slice(l, l + i) for l, i in zip(lo_corner,
                                                          input_shape)]
                subraw = raw_data[tuple(slices)].astype(np.float32) / 255.0

                slices = [slice(l + o, l + o + i)
                          for l, i, o in zip(lo_corner, output_shape,
                                             raw_to_gt_offsets)]
                subgt = (gt_data[tuple(slices)] > 0).astype(np.float32)
                subdist = dists[tuple(slices)].astype(np.float32)
                subgt[(subdist <= ignore_distance) & (subgt == 0)] = 0.5

                # make sure we have enough positive pixels
                if subgt.mean() > 0.015:
                    break

            # flips
            if np.random.randint(2) == 1:
                subraw = subraw[::-1, :, :]
                subgt = subgt[::-1, :, :]
            if np.random.randint(2) == 1:
                subraw = subraw[:, ::-1, :]
                subgt = subgt[:, ::-1, :]
            if np.random.randint(2) == 1:
                subraw = subraw[:, :, ::-1]
                subgt = subgt[:, :, ::-1]
            if np.random.randint(2) == 1:
                subraw = np.transpose(subraw, [0, 2, 1])
                subgt = np.transpose(subgt, [0, 2, 1])

            # random scale/shift of intensities
            scale = np.random.uniform(0.8, 1.2)
            offset = np.random.uniform(-0.2, .2)
            subraw = subraw * scale + offset

            batch.append((subraw, subgt))

        subraws, subgts = zip(*batch)

        # after stack, channel index shifts over one
        yield (np.expand_dims(np.stack(subraws, axis=0), channel_idx + 1),
               np.expand_dims(np.stack(subgts, axis=0), channel_idx + 1))

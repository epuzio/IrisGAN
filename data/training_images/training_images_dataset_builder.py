"""training_images dataset."""

import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for training_images dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(192, 256, 3)),
            # 'label': tfds.features.ClassLabel(names=['no', 'yes']),
        }),
    #     # If there's a common (input, target) tuple from the
    #     # features, specify them here. They'll be used if
    #     # `as_supervised=True` in `builder.as_dataset`.
    #     supervised_keys=('image', 'label'),  # Set to `None` to disable
    #     homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(training_images): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://todo-data-url')

    # TODO(training_images): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'train_imgs'),
        'test': self._generate_examples(path / 'test_images')
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(training_images): Yields (key, example) tuples from the dataset
    for f in path.glob('*.jpeg'):
      yield 'key', {
          'image': f,
          'label': 'yes',
      }

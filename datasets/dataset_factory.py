"""A factory-pattern class which returns classification image/label pairs."""
from datasets import dataset_utils
from datasets.data_configs import *
from DSB2018_get_dataset import get_split as DSB2018GetDataset
from ISBI2017_get_dataset import get_split as ISBI2017GetDataset
from ISBI2017_get_dataset_V2 import get_split as ISBI2017GetDatasetV2

datasets_map = {
    'icdar2013':icdar2013,
    'icdar2015':icdar2015,
    'scut':scut,
    'td500':td500,
    'tr400':tr400,
    'synthtext':synthtext,
    'dsb2018': dsb2018,
    'isbi2017': isbi2017,
    'isbi2017v2': isbi2017
}
resolve_funcs = {
    'dsb2018': DSB2018GetDataset,
    'isbi2017': ISBI2017GetDataset,
    'isbi2017v2': ISBI2017GetDatasetV2,
    'medicalimage': dataset_utils.get_split
}



def get_dataset(dataset_name, split_name, dataset_dir, reader=None):
    """Given a dataset dataset_name and a split_name returns a Dataset.
    Args:
        dataset_name: String, the dataset_name of the dataset.
        split_name: A train/test split dataset_name.
        dataset_dir: The directory where the dataset files are stored.
        reader: The subclass of tf.ReaderBase. If left as `None`, then the default
            reader defined by each dataset is used.
    Returns:
        A `Dataset` class.
    Raises:
        ValueError: If the dataset `dataset_name` is unknown.
    """
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    dataset_config = datasets_map[dataset_name]
    file_pattern = dataset_config.file_pattern
    num_samples = dataset_config.split_sizes[split_name]
    return resolve_funcs[dataset_name](split_name, dataset_dir, file_pattern, num_samples, reader)

class DatasetConfig():
    def __init__(self, file_pattern, split_sizes):
        self.file_pattern = file_pattern
        self.split_sizes = split_sizes


icdar2013 = DatasetConfig(
        file_pattern = '*_%s.tfrecord',
        split_sizes = {
            'train': 229,
            'test': 233
        }
)
icdar2015 = DatasetConfig(
        file_pattern = 'icdar2015_%s.tfrecord',
        split_sizes = {
            'train': 1000,
            'test': 500
        }
)
td500 = DatasetConfig(
        file_pattern = '*_%s.tfrecord',
        split_sizes = {
            'train': 300,
            'test': 200
        }
)
tr400 = DatasetConfig(
        file_pattern = 'tr400_%s.tfrecord',
        split_sizes = {
            'train': 400
        }
)
scut = DatasetConfig(
    file_pattern = 'scut_%s.tfrecord',
    split_sizes = {
        'train': 1715
    }
)

synthtext = DatasetConfig(
    file_pattern = '*.tfrecord',
    split_sizes = {
        'train': 858750
    }
)

dsb2018 = DatasetConfig(
    file_pattern='DSB2018*.tfrecord',
    split_sizes={
        'train': 670
    }
)

isbi2017 = DatasetConfig(
    file_pattern='ISBI2017*.tfrecord',
    split_sizes={
        'train': 1300
    }
)

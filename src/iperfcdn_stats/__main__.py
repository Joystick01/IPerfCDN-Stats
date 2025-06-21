from pyarrow.parquet import ParquetDataset
# Source is either the filename or an Arrow file handle (which could be on HDFS)

source = "../data/IPv4.parquet"
dataset = ParquetDataset(source)
print(dataset.schema.to_string(truncate_metadata=False))


def main():
    pass
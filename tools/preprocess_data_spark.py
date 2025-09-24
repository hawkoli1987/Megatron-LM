# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import logging

# Option 1: Set higher log level for the specific logger
logging.getLogger('megatron.core.datasets.indexed_dataset').setLevel(logging.WARNING)

"""Processing large data for pretraining."""
import argparse
import json
import os
import sys
from pyspark.sql import SparkSession
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import glob
import multiprocessing

import functools
try:
    import nltk
    from nltk.tokenize.punkt import PunktLanguageVars
    nltk_available = True
except ImportError:
    PunktLanguageVars = object  # Fallback to the built-in object class
    nltk_available = False

from megatron.training.tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args
from megatron.core.datasets import indexed_dataset


def timing_decorator(func):
    """Decorator to measure and print the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"{time.strftime('%H:%M:%S', time.localtime())} Process - {func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper

# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            if os.environ.get("NLTK_DATA"):
                library = os.path.join(os.environ.get("NLTK_DATA"), "tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"file:{library}"
            else:
                library = os.path.join("tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"nltk:{library}"
            splitter = nltk.load(url)
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            tokens_list = [Encoder.splitter.tokenize(text[i:i+max_len]) for i in range(0, len(text), max_len)]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
        return json.dumps(output), len(json_line)

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = data[key]
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line)


class Partition(object):
    @timing_decorator
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    @timing_decorator
    def split_sentences(self, file_name):
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')
        fout = open(output_file_name, 'w')

        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        split_docs = pool.imap(encoder.split, fin, 32)

        proc_start = time.time()
        total_bytes_processed = 0
        for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
            total_bytes_processed += bytes_processed
            fout.write(doc + "\n")
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        fout.close()

    def process_input_file(self, file_name):
        # Unpack input file name and output prefix.
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)
        
        file_open_start = time.time()
        
        if hasattr(self.args, 'workers') and self.args.workers is not None:
            print(f"Using {self.args.workers} workers")
            n_workers = self.args.workers
        # default if not set
        else:
            print("Using default number of workers")
            n_workers = multiprocessing.cpu_count() - 5
        
        # Calculate partition size in MB (max of executor memory / n_workers or 256MB)
        executor_memory_gb = 32
        per_worker_RAM_mb = min(executor_memory_gb * 1024 // n_workers, 256)
        
        print(f"n_workers: {n_workers}")
        print(f"pyspark RAM per worker: {per_worker_RAM_mb}m")

        # Create a SparkSession (or get the existing one).
        spark = SparkSession.builder \
            .master(f"local[{n_workers}]") \
            .config("spark.driver.memory", f"{executor_memory_gb}g") \
            .config("spark.executor.memory", f"{executor_memory_gb}g") \
            .config("spark.sql.files.maxPartitionBytes", f"{per_worker_RAM_mb}m") \
            .getOrCreate()
        sc = spark.sparkContext
        
        # Handle 3 cases for input file processing and convert all to DataFrame RDD:
        # 1. if input_file_name is a single parquet file, read it as parquet file, then convert to RDD of Row objects
        # 2. if input_file_name is a single jsonl file, read it in, and convert it to pyspark dataframe, then convert to RDD of Row objects
        # 3. if input_file_name is a list of jsonl files, load them into a single pyspark dataframe, then convert to RDD of Row objects
        
        # Handle 3 cases for input file processing:
        if input_file_name[0].endswith('.parquet'):
            # Case 1: Single parquet file
            print(f"Loading parquet file: {input_file_name}")
            assert len(input_file_name) == 1, "Expected a single parquet file"
            df = spark.read.parquet(input_file_name[0])
            rdd = df.rdd
        elif input_file_name[0].endswith('.jsonl') or input_file_name[0].endswith('.json'):
            # Case 2&3: JSON files (single or list)
            print(f"Loading {len(input_file_name)} jsonl files into DataFrame")
            df = spark.read.option("multiline", "false").json(input_file_name)
            rdd = df.rdd
        else:
            raise ValueError(f"Unsupported input file type: {input_file_name}. Expected: .jsonl, .json, .parquet files, or a list of jsonl files.")
        
        file_open_end = time.time()
        print(f"{time.strftime('%H:%M:%S', time.localtime())} IN - Opening file took {file_open_end - file_open_start:.2f} seconds")
        
        # Startup phase: initialize a dummy encoder (for driver) and obtain the tokenizer.
        startup_start = time.time()
        encoder_dummy = Encoder(self.args)  # for driver initialization only
        tokenizer = build_tokenizer(self.args)
        
        # Determine processing level.
        level = "document"
        if self.args.split_sentences:
            level = "sentence"
        
        # Process a partition and write out its results
        # in the same bin/idx format using the IndexedDatasetBuilder.
        # Now only handles PySpark DataFrame input (Row objects).
        def process_and_write_partition(index, iterator):
            # Each partition gets its own Encoder instance.
            local_encoder = Encoder(self.args)
            local_encoder.initializer()
            
            # Initialize an IndexedDatasetBuilder for each json key.
            local_builders = {}
            for key in self.args.json_keys:
                bin_file = "{}_{}_{}_{}.bin".format(output_prefix, key, level, index)
                # Note: We create a new builder for each key.
                local_builders[key] = indexed_dataset.IndexedDatasetBuilder(
                    bin_file,
                    dtype=indexed_dataset.DType.optimal_dtype(build_tokenizer(self.args).vocab_size)
                )
            
            # Process each Row object in the partition.
            for row in iterator:
                # Convert PySpark Row object to JSON string for processing
                row_dict = row.asDict()
                json_line = json.dumps(row_dict)
                
                doc, sentence_lens, bytes_processed = local_encoder.encode(json_line)
                for key in doc.keys():
                    local_builders[key].add_document(doc[key], sentence_lens[key])
            
            # Finalize each builder to write out the corresponding idx file.
            for key in local_builders:
                idx_file = "{}_{}_{}_{}.idx".format(output_prefix, key, level, index)
                local_builders[key].finalize(idx_file)
            
            # Return an empty iterator
            return iter([])
        
        rdd.mapPartitionsWithIndex(process_and_write_partition).count()  # Just to trigger execution
        
        startup_end = time.time()
        print(f"{time.strftime('%H:%M:%S', time.localtime())} IN - Partition processing and writing took {startup_end - file_open_end:.2f} seconds")
        
        # ---------------------------
        # Phase 2: Merge the per-partition output files into one final dataset.
        # ---------------------------
        
        # Now, for each JSON key, merge all partition outputs.
        merge_start_time = time.time()
        for key in self.args.json_keys:
            # Final output names.
            output_full_prefix = "{}_{}_{}".format(output_prefix, key, level)
            output_bin_file = "{}.bin".format(output_full_prefix)
            print(f"output_bin_file: {output_bin_file}")
            output_idx_file = "{}.idx".format(output_full_prefix)
            # Initialize the final builder.
            builder = indexed_dataset.IndexedDatasetBuilder(
                output_bin_file,
                dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size)
            )
            # Find all partitions with the common output prefix and extract unique partition prefixes
            partitions = sorted(
                glob.glob(f"{output_prefix}_{key}_{level}_*.idx"),
                key=lambda x: int(x.split('_')[-1].split('.')[0])  # Sort by numeric partition index
            )
            print(f"{time.strftime('%H:%M:%S', time.localtime())} IN - Found {len(partitions)} shards of {output_full_prefix}")
            for partition in partitions:
                # Extract the base name without extension
                partition_name = os.path.splitext(partition)[0]
                builder.add_index(partition_name)
            # Finalize the final builder to merge all indices and write the idx file.
            builder.finalize(output_idx_file)
        
        print(f"{time.strftime('%H:%M:%S', time.localtime())} IN - Merging shards completed in {time.time() - merge_start_time:.2f} seconds")
        
        # ---------------------------
        # Phase 3: Clean up intermediate files
        # ---------------------------
        cleanup_start = time.time()
        deleted_count = 0
        
        # Delete all partition .idx files
        for idx_file in glob.glob(f"{output_prefix}_*_{level}_*.idx"):
            os.remove(idx_file)
            deleted_count += 1
            
        # Delete all partition .bin files 
        for bin_file in glob.glob(f"{output_prefix}_*_{level}_*.bin"):
            os.remove(bin_file)
            deleted_count += 1
        
        cleanup_end = time.time()
        print(f"{time.strftime('%H:%M:%S', time.localtime())} IN - Deleted {deleted_count} intermediate files in {cleanup_end - cleanup_start:.2f} seconds")

def get_args():
    parser = argparse.ArgumentParser()
    parser = _add_tokenizer_args(parser)
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON, or a flattened list of paths')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')
    group = parser.add_argument_group(title='tokenization process')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=None, required=False,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--partitions', type=int, default=1,
                        help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is '
                            'preserved when using partitions>1.')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

@timing_decorator
def check_files_exist(in_ss_out_names, key, num_partitions):
    """Checks if all partition files for a given output directory and key exist."""
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True


def main():
    args = get_args()

    # TODO: remove sentence split support in future
    if args.split_sentences:
        if nltk_available:
            nltk.download("punkt", quiet=True, download_dir=os.environ.get("NLTK_DATA"))
        else:
            raise Exception(
                "nltk library required for sentence splitting is not available.")

    in_ss_out_names = []

    print(f"{time.strftime('%H:%M:%S', time.localtime())}  Sentence splitting is {'enabled' if args.split_sentences else 'disabled'}")

    args.input = args.input.split()
    
    # TODO: remove sentence split support in future
    # get a single filename
    representative_file_name, extension = os.path.splitext(args.input[0])
    sentence_split_file = representative_file_name + "_ss" + extension
    in_ss_out_name = {
        'partition': args.input, # can be a flattened list of jsonl filepath, or single jsonl/parquet filepath
        'sentence_split': sentence_split_file,
        'output_prefix': args.output_prefix # a single representativeoutput prefix even the input is a list
        }
    in_ss_out_names.append(in_ss_out_name)

    partition = Partition(args, args.workers)

    # TODO: remove sentence split support in future
    # check to see if paritions with split sentences already created
    split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

    # TODO: remove sentence split support in future
    # split sentences in partition files
    if args.split_sentences and not split_sentences_present:
        for in_ss_out_name in in_ss_out_names:
            partition.split_sentences((in_ss_out_name['partition'], in_ss_out_name['sentence_split']))
        return

    # encode partition files in parallel
    input_key = 'sentence_split' if args.split_sentences else 'partition'
    
    process_json_start = time.time()
    # it applies to both single and multiple partitions
    for in_ss_out_name in in_ss_out_names:
        partition.process_input_file((in_ss_out_name[input_key], in_ss_out_name['output_prefix']))

    process_json_end = time.time()
    print(f"{time.strftime('%H:%M:%S', time.localtime())}  Process - Process json took {process_json_end - process_json_start:.2f} seconds")

    return

if __name__ == '__main__':

    main()


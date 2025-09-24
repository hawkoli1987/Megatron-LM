# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import tempfile
import time
from pathlib import Path
import nltk
import pandas as pd
import pytest
import requests

from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.group_input_jsonl import input_jsonl_group_by_size
from megatron.training.tokenizer.gpt2_tokenization import (
    PRETRAINED_MERGES_ARCHIVE_MAP,
    PRETRAINED_VOCAB_ARCHIVE_MAP,
)

from tools.merge_datasets import main as merge_main
from tools.preprocess_data_spark import Encoder
from tools.preprocess_data_spark import get_args as build_args
from tools.preprocess_data_spark import main as build_main

__HUGGINGFACE_BERT_BASE_UNCASED_VOCAB = (
    "https://huggingface.co/bert-base-uncased/raw/main/vocab.txt"
)

__LOCAL_BERT_VOCAB = "/home/gitlab-runner/data/bert_data/vocab.txt"

__LOCAL_GPT2_MERGE = "/home/gitlab-runner/data/gpt3_data/gpt2-merges.txt"

__LOCAL_GPT2_VOCAB = "/home/gitlab-runner/data/gpt3_data/gpt2-vocab.json"

def dummy_jsonl(odir):
    # numbers
    list_numbers = [json.dumps({"text": str(i + 1)}) + "\n" for i in range(100)]
    with open(os.path.join(odir, "numbers.jsonl"), "w") as writer:
        writer.writelines(list_numbers)
    # numbers ascending
    list_numbers_ascending = [
        json.dumps({"text": " ".join([str(j + 1) for j in range(i + 1)])}) + "\n"
        for i in range(100)
    ]
    with open(os.path.join(odir, "numbers_ascending.jsonl"), "w") as writer:
        writer.writelines(list_numbers_ascending)
    # test
    list_test = []
    with open(__file__) as reader:
        for line in reader:
            list_test.append(json.dumps({"text": line}) + "\n")
    with open(os.path.join(odir, "test.jsonl"), "w") as writer:
        writer.writelines(list_test)

NUM_SAMPLES = 1000
SEQ_LEN = 8192
# print(f"NUM_SAMPLES: {NUM_SAMPLES}")
print(f"SEQ_LEN: {SEQ_LEN}")

def dummy_long_jsonl(odir):
    """
    Creates a jsonl file with NUM_SAMPLES rows, each row containing a sequence of '1' characters
    with length SEQ_LEN.
    
    Args:
        odir: Output directory where the jsonl file will be saved
        seq_len: Length of the sequence of '1' characters in each row
    """
    # Create a sequence of '1' characters with length SEQ_LEN
    sequence = '1' * SEQ_LEN
    
    # Generate NUM_SAMPLES rows, each containing the sequence
    list_sequences = [json.dumps({"text": sequence}) + "\n" for _ in range(NUM_SAMPLES)]
    
    # Write the rows to a jsonl file
    with open(os.path.join(odir, "sequences.jsonl"), "w") as writer:
        writer.writelines(list_sequences)

def dummy_parquet(odir):
    """
    Creates a parquet file with NUM_SAMPLES rows, each row containing a sequence of '1' characters
    with length SEQ_LEN. Target size: ~100GB.
    
    Args:
        odir: Output directory where the parquet file will be saved
    """
    # Create a sequence of '1' characters with length SEQ_LEN
    sequence = '1' * SEQ_LEN
    
    samples_per_file = max(1, 1073741824 // SEQ_LEN)  # 1GB / SEQ_LEN

    # Generate NUM_SAMPLES rows, each containing the sequence
    data = [{"text": sequence} for _ in range(samples_per_file)]
    
    # Create DataFrame and save as parquet
    df = pd.DataFrame(data)
    parquet_path = os.path.join(odir, "sequences.parquet")
    df.to_parquet(parquet_path, index=False)
    
    # Print size of parquet file
    parquet_size = os.path.getsize(parquet_path)
    parquet_size_gb = parquet_size / (1024**3)
    print(f"Parquet file size: {parquet_size:,} bytes ({parquet_size_gb:.2f} GB)", flush=True)

def dummy_jsonl_list(odir):
    """
    Creates 100 jsonl files (0.jsonl to 99.jsonl), each approximately 1GB in size.
    Each file contains sequences of '1' characters.
    
    Args:
        odir: Output directory where the jsonl files will be saved
    """
    # Calculate samples per file to achieve ~1GB per file
    # Assuming each character is 1 byte, and we want 1GB = 1,073,741,824 bytes
    # With SEQ_LEN characters per sample, we need: 1,073,741,824 / SEQ_LEN samples per file
    samples_per_file = max(1, 10737418 // SEQ_LEN)  # 0.01GB / SEQ_LEN
    
    # Create 100 files (0.jsonl to 99.jsonl)
    for i in range(100):
        filename = f"{i}.jsonl"
        filepath = os.path.join(odir, filename)
        
        # Create a sequence of '1' characters with length SEQ_LEN
        sequence = '1' * SEQ_LEN
        
        # Generate samples_per_file rows for this file
        list_sequences = [json.dumps({"text": sequence}) + "\n" for _ in range(samples_per_file)]
        
        # Write the rows to the jsonl file
        with open(filepath, "w") as writer:
            writer.writelines(list_sequences)
        
        print(f"Created {filepath} with size {os.path.getsize(filepath) / 1024 / 1024 / 1024:.2f} GB", flush=True)

    # Calculate and print total size of all JSONL files
    total_size_bytes = 0
    jsonl_files = [f for f in os.listdir(odir) if f.endswith('.jsonl')]
    for filename in jsonl_files:
        filepath = os.path.join(odir, filename)
        total_size_bytes += os.path.getsize(filepath)
    
    total_size_gb = total_size_bytes / (1024**3)
    print(f"Total size of {len(jsonl_files)} JSONL files: {total_size_bytes:,} bytes ({total_size_gb:.2f} GB)", flush=True)
    print("creation of jsonl_list completed", flush=True)

def dummy_EN_DCLM_Edu(odir):
    """
    Creates a jsonl file with EN_DCLM_Edu data.
    """
    input_dir = "/gojek/data/EN/EN_DCLM-Edu"
    # Replace odir with a symlink pointing to input_dir, and confirm odir exists

    import shutil
    shutil.rmtree(odir)
    os.symlink(input_dir, odir)
    assert os.path.exists(odir), f"Symlink {odir} was not created"

def build_datasets(idir, odir, extra_args=[]):
    input_files = [str(f) for f in Path(idir).glob(f"*")]
    # Sort by numeric filename if possible, otherwise alphanumeric
    def sort_key(x):
        stem = Path(x).stem
        if stem.isdigit():
            return (0, int(stem))  # Numeric files first, sorted numerically
        else:
            return (1, stem)  # Alphanumeric files second, sorted alphabetically
    
    input_files = sorted(input_files, key=sort_key)
    print(f"input_files before grouping: {input_files}") # always the fullpath
    if len(input_files) > 1:
        input_files = input_jsonl_group_by_size(input_files, target_size_gb=0.1, mean_threshold_gb=0.05)
    print(f"input_files after grouping: {input_files}") # always the fullpath
    for name in input_files:
        print("-"*100, flush=True)
        print(f"building dataset for: {name}")
        if isinstance(name, list):
            flatten_name = " ".join(name) # concat the list of input filepaths into a single string for input
            representative_name = name[0] # collapse the list of filepaths into a single filepath for output
            sys.argv = [
                sys.argv[0],
                "--input",
                flatten_name,
                "--output-prefix",
                os.path.join(odir, os.path.splitext(os.path.basename(representative_name))[0]), # no extension
            ] + extra_args
        else:
            sys.argv = [
                sys.argv[0],
                "--input",
                name,
                "--output-prefix",
                os.path.join(odir, os.path.splitext(os.path.basename(name))[0]),
            ] + extra_args
        print(f"build dataset sys.argv: {sys.argv}")
        build_main()
        print("-"*100, flush=True)


def merge_datasets(idir):
    """
    Merges multiple pairs of .bin and .idx files in the input directory into a single pair.
    If only one pair exists, creates a duplicate pair named 'merge'.
    """
    sys.argv = [sys.argv[0], "--input", idir, "--output-prefix", os.path.join(idir, "merge")]
    print(f"sys.argv: {sys.argv}")
    merge_main()


def do_test_preprocess_data(temp_dir, extra_args=[]):
    # set the default nltk data path
    os.environ["NLTK_DATA"] = os.path.join(temp_dir, "nltk_data")
    nltk.data.path.append(os.environ["NLTK_DATA"])

    path_to_raws = os.path.join(temp_dir, "sample_raws")
    path_to_data = os.path.join(temp_dir, "sample_data")
    os.mkdir(path_to_data)

    # build the datasets
    start_time = time.time()
    build_datasets(path_to_raws, path_to_data, extra_args=extra_args)
    build_time = time.time() - start_time
    print(f"{time.strftime('%H:%M:%S', time.localtime())}  Test - Dataset building completed in {build_time:.2f} seconds")

    # merge the datasets
    start_time = time.time()
    merge_datasets(path_to_data)
    merge_time = time.time() - start_time
    print(f"{time.strftime('%H:%M:%S', time.localtime())}  Test - Dataset merging completed in {merge_time:.2f} seconds")

    # Print size of merged dataset
    merge_bin_path = os.path.join(path_to_data, "merge.bin")
    print(f"predicted merge_bin_path: {merge_bin_path}")
    print("Files under path_to_data:")
    for fname in sorted(os.listdir(path_to_data)):
        print("  -", fname)
    
    if os.path.exists(merge_bin_path):
        merge_bin_size = os.path.getsize(merge_bin_path)
        print(f"  - merge.bin: {merge_bin_size / (1024**3):.2f} GB")
    else:
        print("Warning: Merged dataset files not found!")

    sys.argv = [sys.argv[0], "--input", None, "--output-prefix", None] + extra_args
    encoder = Encoder(build_args())
    encoder.initializer()

    def tokens_to_string(toks):
        for option in ["decode", "detokenize"]:
            try:
                return getattr(encoder.tokenizer, option)(toks)
            except:
                continue
        raise RuntimeError(f"{type(encoder.tokenizer)} tokenizer cannot decode or detokenize")

    merged_index = 0
    merged_dataset = IndexedDataset(os.path.join(path_to_data, "merge"))

    # sorted to ensure ordering matches merged dataset
    basenames = sorted(
        [
            name
            for name in os.listdir(path_to_data)
            if name.endswith(".idx") and not name.startswith("merge")
        ]
    )

    # index into the merged document index
    merged_doc_index_index = 0

    for basename in basenames:
        # skip checking for parquet files and large file numbers given their large size
        if basename.endswith(".parquet"):
            return
        if len(basename) > 10:
            return

        print(f"path_to_raws: {path_to_raws}")
        print(f"basename: {basename}")
        realpath_raw = f"{os.path.join(path_to_raws, '_'.join(basename.split('_')[:-2]))}.jsonl"
        print(f"realpath_raw: {realpath_raw}")

        realpath_doc = os.path.join(path_to_data, basename.split(".")[-2])

        dataset_index = 0
        dataset = IndexedDataset(realpath_doc)

        merged_doc_idx = merged_dataset.document_indices[
            merged_doc_index_index : merged_doc_index_index + len(dataset.document_indices)
        ]
        merged_doc_idx = merged_doc_idx - merged_doc_idx[0]

        assert (
            dataset.document_indices == merged_doc_idx
        ).all(), f"ERROR: {basename.split('_')[:-2]}: merged dataset document indices mismatch"

        merged_doc_index_index += len(dataset.document_indices) - 1

        with open(realpath_raw, "rt") as reader:
            for json_line in reader:
                toks = encoder.encode(json_line)[0]["text"]

                raw = tokens_to_string(toks)

                processed_toks = []
                while len(processed_toks) < len(toks):
                    processed_toks.extend(dataset[dataset_index])
                    dataset_index += 1
                processed = tokens_to_string(processed_toks)

                assert (
                    raw == processed
                ), f"ERROR: {basename.split('_')[:-2]}: raw and processed documents do not match"

                merged_toks = []
                while len(merged_toks) < len(toks):
                    merged_toks.extend(merged_dataset[merged_index])
                    merged_index += 1
                merged = tokens_to_string(merged_toks)

                assert (
                    raw == merged
                ), f"ERROR: {basename.split('_')[:-2]}: raw and merged documents do not match"

        print(
            f"INFO: {''.join(basename.split('_')[:-2])}: raw, processed, and merged documents match!"
        )

    print("INFO: Success!")


def gpt2_vocab(odir):
    if os.path.exists(__LOCAL_GPT2_VOCAB):
        return __LOCAL_GPT2_VOCAB
    path = os.path.join(odir, "vocab.json")
    with open(path, "wb") as writer:
        writer.write(requests.get(PRETRAINED_VOCAB_ARCHIVE_MAP['gpt2']).content)
    return path


def gpt2_merge(odir):
    if os.path.exists(__LOCAL_GPT2_MERGE):
        return __LOCAL_GPT2_MERGE
    path = os.path.join(odir, "merge.txt")
    with open(path, "wb") as writer:
        writer.write(requests.get(PRETRAINED_MERGES_ARCHIVE_MAP['gpt2']).content)
    return path


def test_preprocess_data_gpt(dummy_input_type: str):
    # Set default environment variables if not already set
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
        
    test_start = time.time()
    print(f"{time.strftime('%H:%M:%S', time.localtime())}  Test - Started")
    
    with tempfile.TemporaryDirectory() as temp_dir:

        # gpt specific args
        gpt_args = [
            "--tokenizer-type",
            "GPT2BPETokenizer",
            "--vocab-file",
            gpt2_vocab(temp_dir),
            "--merge-file",
            gpt2_merge(temp_dir),
            "--append-eod",
            # "--workers",
            # "10",
            "--log-interval",
            "1", 
        ]
        print(f"temp_dir: {temp_dir}")

        path_to_raws = os.path.join(temp_dir, "sample_raws")
        os.mkdir(path_to_raws)

        # Create dummy data using dynamic function naming
        start_time = time.time()
        globals()[f"dummy_{dummy_input_type}"](path_to_raws)
        create_time = time.time() - start_time
        print(f"{time.strftime('%H:%M:%S', time.localtime())}  Test - Dataset creation completed in {create_time:.2f} seconds")

        do_test_preprocess_data(temp_dir, extra_args=gpt_args)
    
    test_end = time.time()
    print(f"{time.strftime('%H:%M:%S', time.localtime())}  Test - Total test execution took {test_end - test_start:.2f} seconds")


def bert_vocab(odir):
    if os.path.exists(__LOCAL_BERT_VOCAB):
        return __LOCAL_BERT_VOCAB
    path = os.path.join(odir, "vocab.txt")
    with open(path, "wb") as writer:
        writer.write(requests.get(__HUGGINGFACE_BERT_BASE_UNCASED_VOCAB).content)
    return path

if __name__ == "__main__":
    # test_preprocess_data_gpt(dummy_input_type="jsonl")
    # test_preprocess_data_gpt(dummy_input_type="long_jsonl")
    # test_preprocess_data_gpt(dummy_input_type="parquet")
    test_preprocess_data_gpt(dummy_input_type="jsonl_list")
    # test_preprocess_data_gpt(dummy_input_type="EN_DCLM_Edu")
    # test_preprocess_data_bert()

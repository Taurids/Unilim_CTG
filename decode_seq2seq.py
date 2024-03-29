# coding=utf-8
# The MIT License (MIT)

# Copyright (c) Microsoft Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import glob
import argparse
import math
import random
from tqdm import tqdm, trange
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from tokenization_unilm import UnilmTokenizer, WhitespaceTokenizer

from modeling_unilm import UnilmForSeq2SeqDecode, UnilmConfig
import utils_seq2seq

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (UnilmConfig,)), ())
MODEL_CLASSES = {
    'unilm': (UnilmConfig, UnilmForSeq2SeqDecode, UnilmTokenizer)
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
my_logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def main():
    my_parser = argparse.ArgumentParser()

    # Required parameters
    my_parser.add_argument("--model_type", default=None, type=str, required=True,
                           help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    my_parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                           help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                               ALL_MODELS))
    my_parser.add_argument("--model_recover_path", default=None, type=str,
                           help="The file of fine-tuned pretraining model.")
    my_parser.add_argument("--config_name", default="", type=str,
                           help="Pretrained config name or path if not the same as model_name")
    my_parser.add_argument("--tokenizer_name", default="", type=str,
                           help="Pretrained tokenizer name or path if not the same as model_name")
    my_parser.add_argument("--max_seq_length", default=512, type=int,
                           help="The maximum total input sequence length after WordPiece tokenization. \n"
                                "Sequences longer than this will be truncated, and sequences shorter \n"
                                "than this will be padded.")

    # decoding parameters
    my_parser.add_argument("--input_file", type=str, help="Input file")
    my_parser.add_argument('--subset', type=int, default=0,
                           help="Decode a subset of the input dataset.")
    my_parser.add_argument("--output_file", type=str, help="output file")
    my_parser.add_argument("--split", type=str, default="",
                           help="Data split (train/val/test).")
    my_parser.add_argument('--tokenized_input', action='store_true',
                           help="Whether the input is tokenized.")
    my_parser.add_argument('--seed', type=int, default=7,
                           help="random seed for initialization")
    my_parser.add_argument("--do_lower_case", action='store_true',
                           help="Set this flag if you are using an uncased model.")
    my_parser.add_argument('--batch_size', type=int, default=4,
                           help="Batch size for decoding.")
    my_parser.add_argument('--beam_size', type=int, default=1,
                           help="Beam size for searching")
    my_parser.add_argument('--length_penalty', type=float, default=0,
                           help="Length penalty for beam search")
    my_parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    my_parser.add_argument('--forbid_ignore_word', type=str, default=None,
                           help="Forbid the word during forbid_duplicate_ngrams")
    my_parser.add_argument("--min_len", default=None, type=int)
    my_parser.add_argument('--need_score_traces', action='store_true')
    my_parser.add_argument('--ngram_size', type=int, default=3)
    my_parser.add_argument('--max_tgt_length', type=int, default=64,
                           help="maximum length of target sequence")

    args = my_parser.parse_args()

    if args.need_score_traces and args.beam_size <= 1:
        raise ValueError(
            "Score trace is only available for beam search with beam size > 1.")
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          max_position_embeddings=args.max_seq_length)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    bi_uni_pipeline = []
    bi_uni_pipeline.append(
        utils_seq2seq.Preprocess4Seq2seqDecode(list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids,
                                               max_passage_length=20, max_tgt_length=66, max_ctrl_length=7,
                                               tokenizer=tokenizer))

    test_dataset = utils_seq2seq.Seq2SeqDataset(args.input_file, args.batch_size, tokenizer, args.max_seq_length,
                                                bi_uni_pipeline=bi_uni_pipeline)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  collate_fn=utils_seq2seq.batch_list_to_batch_tensors,
                                                  pin_memory=False)
    # Prepare model
    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]", "[S2S_SOS]"])
    forbid_ignore_set = None
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        print('w_list:', w_list)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))
    for model_recover_path in glob.glob(args.model_recover_path.strip()):
        my_logger.info("***** Recover model: %s *****", model_recover_path)
        model_recover = torch.load(model_recover_path)
        model = model_class.from_pretrained(args.model_name_or_path, state_dict=model_recover, config=config,
                                            mask_word_id=mask_word_id, search_beam_size=args.beam_size,
                                            length_penalty=args.length_penalty,
                                            eos_id=eos_word_ids, sos_id=sos_word_id,
                                            forbid_duplicate_ngrams=args.forbid_duplicate_ngrams,
                                            forbid_ignore_set=forbid_ignore_set, ngram_size=args.ngram_size,
                                            min_len=args.min_len)
        del model_recover
        model.load_state_dict(torch.load('./output_dir/model.bin'))
        model.to(device)

        torch.cuda.empty_cache()
        model.eval()

        output_lines = []
        score_trace_list = []
        iter_bar = tqdm(test_dataloader)
        for step, batch in enumerate(iter_bar):
            with torch.no_grad():
                batch = [t.to(device) if t is not None else None for t in batch]
                input_ids, segment_ids, ctrl_ids, position_ids, input_mask = batch
                traces = model(input_ids, segment_ids, ctrl_ids, position_ids, input_mask)
                if args.beam_size > 1:
                    traces = {k: v.tolist() for k, v in traces.items()}
                    output_ids = traces['pred_seq']
                else:
                    output_ids = traces.tolist()
                    # print('output_ids:', output_ids)
                for i in range(len(input_ids)):
                    w_ids = output_ids[i]
                    output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                    output_tokens = []
                    for t in output_buf:
                        if t in ("[SEP]", "[PAD]"):
                            break
                        output_tokens.append(t)
                    output_sequence = ''.join(detokenize(output_tokens))

                    input_id_decode = tokenizer.decode(input_ids[i])
                    CLS_index = input_id_decode.index('[CLS]')
                    SEP_index = input_id_decode.index('[SEP]')
                    print('输入：', input_id_decode[CLS_index + 6:SEP_index])
                    print('输出：', output_sequence)

                    output_lines.append(output_sequence)
                    if args.need_score_traces:
                        score_trace_list.append(
                            {'scores': traces['scores'][i], 'wids': traces['wids'][i], 'ptrs': traces['ptrs'][i]})


if __name__ == "__main__":
    main()

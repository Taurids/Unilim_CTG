# coding=utf-8
import logging
import glob
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import RandomSampler

from tokenization_unilm import UnilmTokenizer, WhitespaceTokenizer
from modeling_unilm import UnilmForSeq2Seq, UnilmConfig
from pytorch_transformers import AdamW, get_linear_schedule_with_warmup

import utils_seq2seq

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (UnilmConfig,)), ())
MODEL_CLASSES = {
    'unilm': (UnilmConfig, UnilmForSeq2Seq, UnilmTokenizer)
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
my_logger = logging.getLogger(__name__)


def main():
    my_parser = argparse.ArgumentParser()

    # Required parameters
    my_parser.add_argument("--data_dir", default=None, type=str, required=True,
                           help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    my_parser.add_argument("--src_file", default=None, type=str,
                           help="The input data file name.")
    my_parser.add_argument("--model_type", default=None, type=str, required=True,
                           help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    my_parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                           help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                               ALL_MODELS))
    my_parser.add_argument("--output_dir", default=None, type=str, required=True,
                           help="The output directory where the model predictions and checkpoints will be written.")
    my_parser.add_argument("--log_dir", default='', type=str,
                           help="The output directory where the log will be written.")
    my_parser.add_argument("--model_recover_path", default=None, type=str,
                           help="The file of fine-tuned pretraining model.")
    my_parser.add_argument("--optim_recover_path", default=None, type=str,
                           help="The file of pretraining optimizer.")
    my_parser.add_argument("--config_name", default="", type=str,
                           help="Pretrained config name or path if not the same as model_name")
    my_parser.add_argument("--tokenizer_name", default="", type=str,
                           help="Pretrained tokenizer name or path if not the same as model_name")

    # Other parameters
    my_parser.add_argument("--max_seq_length", default=64, type=int,
                           help="The maximum total input sequence length after WordPiece tokenization. \n"
                                "Sequences longer than this will be truncated, and sequences shorter \n"
                                "than this will be padded.")
    my_parser.add_argument('--max_position_embeddings', type=int, default=None,
                           help="max position embeddings")
    my_parser.add_argument("--do_train", action='store_true',
                           help="Whether to run training.")
    my_parser.add_argument("--do_eval", action='store_true',
                           help="Whether to run eval on the dev set.")
    my_parser.add_argument("--do_lower_case", action='store_true',
                           help="Set this flag if you are using an uncased model.")
    my_parser.add_argument("--train_batch_size", default=32, type=int,
                           help="Total batch size for training.")
    my_parser.add_argument("--eval_batch_size", default=32, type=int,
                           help="Total batch size for eval.")
    my_parser.add_argument("--learning_rate", default=5e-5, type=float,
                           help="The initial learning rate for Adam.")
    my_parser.add_argument("--label_smoothing", default=0.1, type=float,
                           help="The initial learning rate for Adam.")
    my_parser.add_argument("--weight_decay", default=0.01, type=float,
                           help="The weight decay rate for Adam.")
    my_parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                           help="Epsilon for Adam optimizer.")
    my_parser.add_argument("--max_grad_norm", default=1.0, type=float,
                           help="Max gradient norm.")
    my_parser.add_argument("--num_train_epochs", default=3.0, type=float,
                           help="Total number of training epochs to perform.")
    my_parser.add_argument("--warmup_proportion", default=0.1, type=float,
                           help="Proportion of training to perform linear learning rate warmup for. "
                                "E.g., 0.1 = 10%% of training.")
    my_parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                           help="Dropout rate for hidden states.")
    my_parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
                           help="Dropout rate for attention probabilities.")
    my_parser.add_argument('--seed', type=int, default=7,
                           help="random seed for initialization")
    my_parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                           help="Number of updates steps to accumulate before performing a backward/update pass.")
    my_parser.add_argument('--tokenized_input', action='store_true',
                           help="Whether the input is tokenized.")
    my_parser.add_argument('--max_len_a', type=int, default=0,
                           help="Truncate_config: maximum length of segment A.")
    my_parser.add_argument('--max_len_b', type=int, default=0,
                           help="Truncate_config: maximum length of segment B.")
    my_parser.add_argument('--trunc_seg', default='',
                           help="Truncate_config: first truncate segment A/B (option: a, b).")
    my_parser.add_argument('--always_truncate_tail', action='store_true',
                           help="Truncate_config: Whether we should always truncate tail.")
    my_parser.add_argument("--mask_prob", default=0.20, type=float,
                           help="Number of prediction is sometimes less than max_pred when sequence is short.")
    my_parser.add_argument("--mask_prob_eos", default=0, type=float,
                           help="Number of prediction is sometimes less than max_pred when sequence is short.")
    my_parser.add_argument('--max_pred', type=int, default=94,
                           help="Max tokens of prediction.")
    my_parser.add_argument("--num_workers", default=0, type=int,
                           help="Number of workers for the data loader.")
    my_parser.add_argument('--mask_source_words', action='store_true',
                           help="Whether to mask source words for training")
    my_parser.add_argument('--skipgram_prb', type=float, default=0.0,
                           help='prob of ngram mask')
    my_parser.add_argument('--skipgram_size', type=int, default=1,
                           help='the max size of ngram mask')
    my_parser.add_argument('--mask_whole_word', action='store_true',
                           help="Whether masking a whole word.")

    args = my_parser.parse_args()

    if not (args.model_recover_path and Path(args.model_recover_path).exists()):
        args.model_recover_path = None

    args.output_dir = args.output_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))
    args.log_dir = args.log_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))

    os.makedirs(args.output_dir, exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

    my_logger.info("device: {}".format(device))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        max_position_embeddings=args.max_position_embeddings,
        label_smoothing=args.label_smoothing,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer

    if args.do_train:
        print("Loading Train Dataset", args.data_dir)
        bi_uni_pipeline = [utils_seq2seq.Preprocess4Seq2seq(args.mask_prob, list(tokenizer.vocab.keys()),
                                                            tokenizer.convert_tokens_to_ids, max_passage_length=20,
                                                            max_tgt_length=66, max_ctrl_length=7,
                                                            mask_source_words=False, skipgram_prb=args.skipgram_prb,
                                                            skipgram_size=args.skipgram_size,
                                                            mask_whole_word=args.mask_whole_word,
                                                            tokenizer=data_tokenizer)]

        file = os.path.join(args.data_dir, args.src_file if args.src_file else 'train.tgt')
        train_dataset = utils_seq2seq.Seq2SeqDataset(
            file, args.train_batch_size, data_tokenizer, args.max_seq_length, bi_uni_pipeline=bi_uni_pipeline)
        train_sampler = RandomSampler(train_dataset, replacement=False)
        _batch_size = args.train_batch_size

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, sampler=train_sampler,
                                                       num_workers=args.num_workers,
                                                       collate_fn=utils_seq2seq.batch_list_to_batch_tensors,
                                                       pin_memory=False)
        print("Loading dev dataset")
        dev_file = os.path.join(args.data_dir, 'dev_data.json')
        dev_dataset = utils_seq2seq.Seq2SeqDataset(dev_file, args.eval_batch_size, data_tokenizer, args.max_seq_length,
                                                   bi_uni_pipeline=bi_uni_pipeline)
        dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.eval_batch_size,
                                                     collate_fn=utils_seq2seq.batch_list_to_batch_tensors,
                                                     pin_memory=False, num_workers=args.num_workers)

    # note: args.train_batch_size has been changed to (/= args.gradient_accumulation_steps)
    # t_total = int(math.ceil(len(train_dataset.ex_list) / args.train_batch_size)
    t_total = int(len(train_dataloader) * args.num_train_epochs / args.gradient_accumulation_steps)

    # Prepare model
    if args.model_recover_path is None:
        model_recover = None
    else:
        my_logger.info("***** Recover model: %s *****", args.model_recover_path)
        model_recover = torch.load(args.model_recover_path, map_location='cpu')
    model = model_class.from_pretrained(args.model_name_or_path, state_dict=model_recover, config=config)
    # model.load_state_dict(torch.load('./output_dir/model.bin'))
    model.to(device)

    my_logger.info("device: {}".format(device))

    # Prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)  # 选择优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.warmup_proportion)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=2e-06)  # 余弦退火

    if args.optim_recover_path:
        if os.path.exists(os.path.join(args.output_dir, "optim.bin")):
            my_logger.info("***** Recover optimizer *****")
            optim_recover = torch.load(os.path.join(args.output_dir, "optim.bin"), map_location='cpu')
            if hasattr(optim_recover, 'state_dict'):
                optim_recover = optim_recover.state_dict()
            optimizer.load_state_dict(optim_recover)

            my_logger.info("***** Recover scheduler *****")
            scheduler_recover = torch.load(os.path.join(args.output_dir, "sched.bin"), map_location='cpu')
            scheduler.load_state_dict(scheduler_recover)

    my_logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    if args.do_train:
        my_logger.info("***** Running training *****")
        my_logger.info("  Batch size = %d", args.train_batch_size)
        my_logger.info("  Num steps = %d", t_total)

        model.train()
        start_epoch = 1
        for i_epoch in trange(start_epoch, int(args.num_train_epochs) + 1, desc="Epoch"):
            iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
            final_loss = 0
            for step, batch in enumerate(iter_bar):
                batch = [t.to(device) if t is not None else None for t in batch]
                input_ids, segment_ids, ctrl_ids, input_mask, lm_label_ids, masked_pos, masked_weights, _ = batch
                masked_lm_loss = model(input_ids, segment_ids, ctrl_ids, input_mask, lm_label_ids,
                                       masked_pos=masked_pos, masked_weights=masked_weights)
                loss = masked_lm_loss
                final_loss = loss.item()

                # logging for each step (i.e., before normalization by args.gradient_accumulation_steps)
                iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())

                # ensure that accumlated gradients are normalized
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
            # Save a trained model
            my_logger.info("** ** * Saving fine-tuned model and optimizer ** ** * ")
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, "model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

            output_optim_file = os.path.join(args.output_dir, "optim.bin")
            torch.save(optimizer.state_dict(), output_optim_file)
            output_sched_file = os.path.join(args.output_dir, "sched.bin")
            torch.save(scheduler.state_dict(), output_sched_file)

            my_logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

            if args.do_eval:
                # do_eval
                iter_dev = tqdm(dev_dataloader, desc='Iter (loss=X.XXX)')
                val_losses = []
                for step, batch in enumerate(iter_dev):
                    with torch.no_grad():
                        batch = [t.to(device) if t is not None else None for t in batch]
                        input_ids, segment_ids, ctrl_ids, input_mask, lm_label_ids, masked_pos, masked_weights, _ = batch
                        masked_dev_loss = model(input_ids, segment_ids, ctrl_ids, input_mask, lm_label_ids,
                                                masked_pos=masked_pos, masked_weights=masked_weights)
                        val_losses.append(masked_dev_loss.item())
                        iter_dev.set_description('Iter (loss=%5.3f)' % masked_dev_loss.item())

                val_loss = np.mean(val_losses)
                print("Epoch {} - final loss : {:.4f} - val loss :{:.4f}".format(i_epoch, final_loss, val_loss))


if __name__ == "__main__":
    main()

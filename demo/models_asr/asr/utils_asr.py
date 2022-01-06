import torch
import torch.nn.functional as F
import torch.nn as nn
import constant
from helper import get_word_segments_per_language, is_contain_chinese_word
from torch.autograd import Variable
import math

def calculate_metrics(pred, gold, input_lengths=None, target_lengths=None, smoothing=0.0, loss_type="ce"):
    """
    Calculate metrics
    args:
        pred: B x T x C
        gold: B x T
        input_lengths: B (for CTC)
        target_lengths: B (for CTC)
    """
    loss = calculate_loss(pred, gold, input_lengths, target_lengths, smoothing, loss_type)
    if loss_type == "ce":
        pred = pred.view(-1, pred.size(2)) # (B*T) x C
        gold = gold.contiguous().view(-1) # (B*T)
        pred = pred.max(1)[1]
        non_pad_mask = gold.ne(constant.PAD_TOKEN)
        num_correct = pred.eq(gold)
        num_correct = num_correct.masked_select(non_pad_mask).sum().item()
        return loss, num_correct
    elif loss_type == "ctc":
        return loss, None
    else:
        print("loss is not defined")
        return None, None


def calculate_lm_score(seq, lm, id2label):
    """
    seq: (1, seq_len)
    id2label: map
    """
    # print("hello")
    seq_str = "".join(id2label[char.item()] for char in seq[0]).replace(
        constant.PAD_CHAR, "").replace(constant.SOS_CHAR, "").replace(constant.EOS_CHAR, "")
    seq_str = seq_str.replace("  ", " ")

    seq_arr = get_word_segments_per_language(seq_str)
    seq_str = ""
    for i in range(len(seq_arr)):
        if is_contain_chinese_word(seq_arr[i]):
            for char in seq_arr[i]:
                if seq_str != "":
                    seq_str += " "
                seq_str += char
        else:
            if seq_str != "":
                seq_str += " "
            seq_str += seq_arr[i]

    # print("seq_str:", seq_str)
    seq_str = seq_str.replace("  ", " ").replace("  ", " ")
    # print("seq str:", seq_str)

    if seq_str == "":
        return -999, 0, 0

    score, oov_token = lm.evaluate(seq_str)    
    
    # a, b = lm.evaluate("除非 的 不会 improve 什么 东西 的 这些 esperience")
    # a2, b2 = lm.evaluate("除非 的 不会 improve 什么 东西 的 这些 experience")
    # print(a, a2)
    return -1 * score / len(seq_str.split()) + 1, len(seq_str.split()) + 1, oov_token

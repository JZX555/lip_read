# encoder=utf8
import collections
import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import six
import unicodedata
import sys
import re
from hyper_and_conf import conf_fn as conf_train


def token_trim(tokens, trim_id):
    try:
        trim = tokens.index(trim_id)
        tokens = tokens[:trim]
    except Exception:
        pass
    return tokens


def bleu_score(logits, labels, eos_id=1):
    """Approximate BLEU score computation between labels and predictions.
  An approximate BLEU scoring method since we do not glue word pieces or
  decode the ids and tokenize the output. By default, we use ngram order of 4
  and use brevity penalty. Also, this does not have beam search.
  Args:
    logits: Tensor of size [batch_size, length_logits, vocab_size]
    labels: Tensor of size [batch-size, length_labels]
  Returns:
    bleu: int, approx bleu score
  """
    # predictions = tf.to_int32(tf.argmax(logits, axis=-1))
    # TODO: Look into removing use of py_func
    bleu = tf.py_function(compute_bleu, (labels, logits, eos_id), tf.float32)

    return bleu, tf.constant(1.0)


def _get_ngrams_with_counter(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(raw_reference_corpus,
                 raw_translation_corpus,
                 eos_id=1,
                 max_order=4,
                 use_bp=True):
    """Computes BLEU score of translated segments against one or more references.
  Args:
    reference_corpus: list of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    use_bp: boolean, whether to apply brevity penalty.
  Returns:
    BLEU score.
  """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    precisions = []
    try:
        eos_id = eos_id.numpy()
        # eos_id = 1
        reference_corpus = raw_reference_corpus.numpy().tolist()
        translation_corpus = raw_translation_corpus.numpy().tolist()
    except Exception:
        eos_id = eos_id
        # eos_id = 1
        reference_corpus = raw_reference_corpus
        translation_corpus = raw_translation_corpus

    for (references, translations) in zip(reference_corpus,
                                          translation_corpus):
        # references = token_trim(references, eos_id)
        # translations = token_trim(translations, eos_id)
        # references = data_manager.decode(references).split(' ')
        # translations = data_manager.decode(translations).split(' ')
        reference_length += len(references)
        translation_length += len(translations)
        ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
        translation_ngram_counts = _get_ngrams_with_counter(
            translations, max_order)

        overlap = dict((ngram, min(count, translation_ngram_counts[ngram]))
                       for ngram, count in ref_ngram_counts.items())

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram) -
                                      1] += translation_ngram_counts[ngram]

    precisions = [0] * max_order
    smooth = 1.0

    for i in xrange(0, max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = float(
                matches_by_order[i]) / possible_matches_by_order[i]
            if matches_by_order[i] > 0:
                precisions[i] = float(
                    matches_by_order[i]) / possible_matches_by_order[i]
            else:
                smooth *= 2
                precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
        else:
            precisions[i] = 0.0

    if max(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions if p)
        geo_mean = math.exp(p_log_sum / max_order)

    if use_bp:
        ratio = translation_length / reference_length
        if ratio != 0:
            bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
        else:
            bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
    bleu = geo_mean * bp
    return np.float32(bleu)


class UnicodeRegex(object):
    """Ad-hoc hack to recognize all punctuation and symbols."""

    def __init__(self):
        punctuation = self.property_chars("P")
        self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
        self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
        self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

    def property_chars(self, prefix):
        return "".join(
            six.unichr(x) for x in range(sys.maxunicode)
            if unicodedata.category(six.unichr(x)).startswith(prefix))


uregex = UnicodeRegex()


def bleu_tokenize(string):
    r"""Tokenize a string following the official BLEU implementation.
  See https://github.com/moses-smt/mosesdecoder/'
           'blob/master/scripts/generic/mteval-v14.pl#L954-L983
  In our case, the input string is expected to be just one line
  and no HTML entities de-escaping is needed.
  So we just tokenize on punctuation and symbols,
  except when a punctuation is preceded and followed by a digit
  (e.g. a comma/dot as a thousand/decimal separator).
  Note that a numer (e.g. a year) followed by a dot at the end of sentence
  is NOT tokenized,
  i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
  does not match this case (unless we add a space after each sentence).
  However, this error is already in the original mteval-v14.pl
  and we want to be consistent with it.
  Args:
    string: the input string
  Returns:
    a list of tokens
  """
    string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
    string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
    string = uregex.symbol_re.sub(r" \1 ", string)
    return string.split()


def bleu_wrapper(ref_filename, hyp_filename, case_sensitive=False):
    """Compute BLEU for two files (reference and hypothesis translation)."""
    ref_lines = tf.gfile.Open(ref_filename).read().strip().splitlines()
    hyp_lines = tf.gfile.Open(hyp_filename).read().strip().splitlines()
    # ref_lines = ['I like cat']
    # hyp_lines = ['I like dog']

    if len(ref_lines) != len(hyp_lines):
        raise ValueError(
            "Reference and translation files have different number of "
            "lines.")
    if not case_sensitive:
        ref_lines = [x.lower() for x in ref_lines]
        hyp_lines = [x.lower() for x in hyp_lines]
    ref_tokens = [bleu_tokenize(x) for x in ref_lines]
    hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]
    return compute_bleu(ref_tokens, hyp_tokens) * 100


# sess = tf.Session()
# re = tf.convert_to_tensor([[2, 2, 3, 1, 0, 0], [3, 4, 6, 1, 0,
#                                                 0]]).eval(session=sess)
# tr = [[1, 2, 2, 134, 234, 123, 3, 2], [3, 6, 61, 23, 5, 6, 7, 2]]
# s, c = bleu_score(tr, re, 1)
# c.eval(session=sess)
# s.eval(session=sess)
# ref = '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng/europarl-v7.fr-en.en_test'
# tra = '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng/europarl-v7.fr-en.fr_test'
# bleu_wrapper(ref, tra)


def padded_accuracy_score(labels, logits):
    score = tf.py_function(padded_accuracy, (logits, labels), tf.float32)
    return score


def padded_accuracy_score_topk(labels, logits, k=5):
    score = tf.py_function(padded_accuracy_topk, (logits, labels, k),
                           tf.float32)
    return score


def padded_accuracy(logits, labels):
    logits = logits.numpy()
    labels = labels.numpy()
    logits, labels = conf_train.pad_tensors_to_same_length(logits, labels)
    weights = tf.to_float(tf.not_equal(labels, 0))
    outputs = tf.to_int32(tf.argmax(logits, axis=-1))
    padded_labels = tf.to_int32(labels)
    score = tf.to_float(tf.equal(outputs, padded_labels))
    score *= weights
    score = tf.reduce_sum(score) / tf.reduce_sum(weights)
    return score


def padded_accuracy_topk(logits, labels, k):
    logits = logits.numpy()
    labels = labels.numpy()
    k = k.numpy()
    logits, labels = conf_train.pad_tensors_to_same_length(logits, labels)
    weights = tf.to_float(tf.not_equal(labels, 0))
    effective_k = tf.minimum(k, tf.shape(logits)[-1])
    _, outputs = tf.nn.top_k(logits, k=effective_k)
    outputs = tf.to_int32(outputs)
    padded_labels = tf.to_int32(labels)
    padded_labels = tf.expand_dims(padded_labels, axis=-1)
    padded_labels += tf.zeros_like(outputs)  # Pad to same shape.
    same = tf.to_float(tf.equal(outputs, padded_labels))
    same_topk = tf.reduce_sum(same, axis=-1)
    same_topk *= weights
    score = tf.reduce_sum(same_topk) / tf.reduce_sum(weights)
    return score

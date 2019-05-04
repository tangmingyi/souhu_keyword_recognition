# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
from lib.bert import modeling
from lib.bert import optimization
from lib.bert import tokenization
import six
import tensorflow as tf
import re
from lib.bert.match_lay import attention_model
from tensorflow.python import debug as tf_debug
import numpy as np
from lib.bert.myhook import evalute_hook, train_hook, tensor_filter
from utility.data_tool.from_txt_get_data import data_util

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", "D:\\programing_data\\data\\bert_chinese_model\\chinese_L-12_H-768_A-12\\bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", "D:\\programing_data\\data\\bert_chinese_model\\chinese_L-12_H-768_A-12\\vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    # "output_dir", "D:\\programing\\souhumodel\\BERT_core_match",
    "output_dir", "res",
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", "data/souhu/train.txt",
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", "data/souhu/dev.txt",
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", "D:\\programing_data\\data\\bert_chinese_model\\chinese_L-12_H-768_A-12\\bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_bool("reload_data", True, "是否重新加载数据并生成tfrecord文件")
flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 160,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 80,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", True, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 2, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 300,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 10,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")


class SquadExample(object):
    """A single training/test example for simple sequence classification.

       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 emotion_label=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False,
                 label=None,
                 label_mark=None
                 ):
        self.emotion_label = emotion_label
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.label = label
        self.label_mark = label_mark

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 example_id,
                 unique_id,
                 answer_mark,
                 content_mark,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 emotion_label,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 label=None,
                 label_mark=None):
        self.example_id = example_id
        self.answer_mark = answer_mark
        self.content_mark = content_mark
        self.emotion_label = emotion_label
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.label = label
        self.label_mark = label_mark


class sohudata():
    def __init__(self, data, istrain=True):
        paten = re.compile(r"\n+")
        self.newsid = data["newsId"]
        self.title = data['title']
        self.content = self.title + data['content'].strip() + self.title
        self.content = re.sub(paten, " ", self.content)
        self.entity_mon = list()
        if istrain:
            for dic in data['coreEntityEmotions']:
                self.entity_mon.append((dic["entity"].strip(), dic["emotion"].strip()))


def read_squad_examples(input_file, is_training):
    train_data = data_util()
    if is_training:
        souhu_file = "data/souhu/train.txt"
        sub_file = "res/dev_sub.txt"
    else:
        souhu_file = "data/souhu/test.txt"
        sub_file = "res/prediction10.txt"
    train_data.get_souhu_data(souhu_file, is_training)
    train_data.get_sub_res_data(sub_file, is_training, FLAGS.max_query_length)
    if is_training:
        train_data.get_sub_train_data()
    else:
        train_data.get_sub_prediction_data()
    examples = []
    for index, data in enumerate(train_data.train_data):
        # if index == 100:
        #     break
        # data = train_data.train_data[10]
        example = SquadExample(
            qas_id=data.new_id,
            question_text=" ".join(list(("@".join(data.predict)))),
            doc_tokens=" ".join(list(data.content)),
            emotion_label=None,
            orig_answer_text=None,
            start_position=None,
            end_position=None,
            is_impossible=True,
            label=data.label,
            label_mark=data.label_mark
        )
        examples.append(example)

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)  # 剪裁问题

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]
        # 源token存放在example.doc_tokens,子token存放在all_doc_token,这里维护一个互查表，使得给出源token在example.doc_tokens中的索引可查其对应的子token在all_doc_token中的索引。
        tok_to_orig_index = []  # index 对用子token的位置，其值对应原token的索引
        orig_to_tok_index = []  # 其index对应原来的token 其值对应子token在all_doc_token中的所索引
        all_doc_tokens = []  # 所有的子token
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        # max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        max_tokens_for_doc = max_seq_length - max_query_length - 3
        if len(all_doc_tokens) > max_tokens_for_doc:
            all_doc_tokens = all_doc_tokens[:max_tokens_for_doc]
        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.

        # _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        #     "DocSpan", ["start", "length"])
        # doc_spans = []
        # start_offset = 0
        # #基于子token的。
        # while start_offset < len(all_doc_tokens):
        #     length = len(all_doc_tokens) - start_offset
        #     if length > max_tokens_for_doc:
        #         length = max_tokens_for_doc
        #     doc_spans.append(_DocSpan(start=start_offset, length=length))
        #     if start_offset + length == len(all_doc_tokens):
        #         break
        #     start_offset += min(length, doc_stride)

        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        answer_mark = []
        content_mark = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            if token == "@":
                token = "[SEP]"
            tokens.append(token)
            segment_ids.append(0)
            answer_mark.append(1)
        for _ in range(max_query_length + 1 - len(segment_ids)):
            tokens.append("[PAD]")
            segment_ids.append(0)
            answer_mark.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(len(all_doc_tokens)):
            tokens.append(all_doc_tokens[i])
            segment_ids.append(1)
            content_mark.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        content_mark.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            content_mark.append(0)

        # if len(content_mark) != max_seq_length - max_query_length -3:
        #     print("debug")
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(answer_mark) == max_query_length
        assert len(content_mark) == max_seq_length - max_query_length - 2

        start_position = None
        end_position = None
        emotion = None

        if is_training and example.is_impossible:
            start_position = 0
            end_position = 0

        if example_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % (unique_id))
            tf.logging.info("example_index: %s" % (example_index))
            # tf.logging.info("doc_span_index: %s" % (doc_span_index))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("token_to_orig_map: %s" % " ".join(
                ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
            tf.logging.info("token_is_max_context: %s" % " ".join([
                "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
            ]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info(
                "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if is_training and example.is_impossible:
                tf.logging.info("impossible example")
            if is_training and not example.is_impossible:
                answer_text = " ".join(
                    tokens[start_position + max_query_length + 2:(end_position + 1 + max_query_length + 2)])
                tf.logging.info("start_position: %d" % (start_position))
                tf.logging.info("end_position: %d" % (end_position))
                tf.logging.info(
                    "answer: %s" % (tokenization.printable_text(answer_text)))
        if is_training:
            emotion = example.emotion_label

        feature = InputFeatures(
            example_id=example.qas_id,
            unique_id=unique_id,
            answer_mark=answer_mark,
            content_mark=content_mark,
            example_index=example_index,
            doc_span_index=None,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            emotion_label=emotion,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position,
            is_impossible=example.is_impossible,
            label=example.label,
            label_mark=example.label_mark)

        # Run callback
        output_fn(feature)

        unique_id += 1


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def create_model(bert_config, is_training, input_ids,
                 input_mask, segment_ids, answer_mark,
                 content_mark, label_mark,
                 use_one_hot_embeddings, answer_max):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()

    answer_vocter = final_hidden[:, 1:answer_max + 1, :]
    content_vocter = final_hidden[:, answer_max + 2:, :]
    all_final_hidden_matrix = []
    batch_size = None
    seq_length = None
    hidden_size = None
    ffc_wight = {}

    def get_variable(name, input_size=bert_config.hidden_size, output_size=bert_config.hidden_size, is_weight=True):
        init = None
        if is_weight:
            init = tf.truncated_normal_initializer(stddev=0.02)
            shape = [output_size, input_size]
        else:
            init = tf.zeros_initializer()
            shape = [output_size]
        return tf.get_variable(name,
                               shape=shape,
                               initializer=init)

    ffc_wight["reper_weight"] = tf.get_variable("tmy/repertation_weight",
                                                shape=[bert_config.hidden_size, 2 * bert_config.hidden_size],
                                                initializer=tf.truncated_normal_initializer(stddev=0.02))
    ffc_wight["reper_bias"] = tf.get_variable("tmy/repertation_bias", shape=[bert_config.hidden_size],
                                              initializer=tf.zeros_initializer())
    ffc_wight["attention"] = {
        "query_weight": get_variable("tmy/attention/query_weight"),
        "query_bias": get_variable("tmy/attention/query_bias",is_weight=False),
        "key_weight": get_variable("tmy/attention/key_weight"),
        "key_bias":get_variable("tmy/attention/key_bias",is_weight=False),
        "value_weight":get_variable("tmy/attention/value_weight"),
        "value_bias":get_variable("tmy/attention/value_bias",is_weight=False)
                            }
    for i in range(20):
        with tf.variable_scope("answer{}".format(i)):
            one_label_mark = label_mark[:, i, :]
            attention_mark = tf.multiply(tf.cast(tf.expand_dims(content_mark, axis=[2]),dtype=tf.float32),
                                         tf.expand_dims(one_label_mark, axis=[1]))
            match_lay = attention_model(content_vocter, answer_vocter,
                                        attention_mark, bert_config.hidden_size,
                                        bert_config.num_attention_heads,
                                        ffc_wight=ffc_wight
                                        )
            representation = match_lay.get_repersentation_vator()

            # final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
            # representation_shape = tf.shape(representation)
            batch_size = tf.shape(representation)[0]
            seq_length = representation.shape[1].value
            hidden_size = representation.shape[2].value

            final_hidden_matrix = tf.squeeze(
                tf.nn.max_pool(tf.expand_dims(representation, axis=-1), ksize=[1, seq_length, 1, 1], padding="VALID",
                               strides=[1, 1, 1, 1], name="max_pool"),axis=[1,-1])
            all_final_hidden_matrix.append(final_hidden_matrix)
    # final_hidden_matrix = tf.reshape(representation,
    #                                  [batch_size * seq_length, hidden_size])
    output_weights = tf.get_variable(
        "cls/squad/output_weights", [2, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())
    all_logits = []
    for index, input_hidden in enumerate(all_final_hidden_matrix):
        with tf.variable_scope("output{}".format(index)):
            logits = tf.matmul(input_hidden, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            all_logits.append(logits)
    return all_logits


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, answer_max, all_max):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        # train_data_set = my_input_fn_builder("res/train_124728_12472_end.tf_record", FLAGS.max_seq_length, True, True)(
        #     FLAGS.train_batch_size)
        # test_data_set = my_input_fn_builder("res/dev.tf_record", FLAGS.max_seq_length, True, True)(FLAGS.train_batch_size)
        # train_iterator = train_data_set.make_initializable_iterator( )
        # test_iterator = test_data_set.make_initializable_iterator()
        # train_handle = tf.Session().run(train_iterator.string_handle())
        # test_handle = tf.Session().run(test_iterator.string_handle())
        # handle = tf.placeholder(tf.string,shape=[])
        # iterator = tf.data.Iterator.from_string_handle(handle,train_iterator.output_types,train_iterator.output_shapes,train_iterator.output_classes)
        # features = iterator.get_next()

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        answer_mark = features["answer_mark"]
        content_mark = features["content_mark"]
        label_mark = tf.reshape(features["label_mark"], shape=[-1, 20, answer_max])
        # attention_mark = tf.multiply(tf.expand_dims(content_mark, axis=[2]), tf.expand_dims(answer_mark, axis=[1]))
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        all_logits = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
            answer_mark=answer_mark,
            content_mark=content_mark,
            label_mark=label_mark,
            answer_max=answer_max)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_length = modeling.get_shape_list(input_ids)[1]

            def compute_loss(logits, positions):
                one_hot_positions = positions
                log_probs = tf.nn.log_softmax(logits, axis=-1)  # todo 为什么使用logsoftmax()???
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                return loss

            label = features["label"]
            batch_size = tf.shape(label)[0]
            label = tf.reshape(label, shape=[batch_size, 20, 2])
            all_loss = []
            for i in range(20):
                true_label = tf.squeeze(label[:, i, :])
                # p = tf.equal(true_label, tf.ones(shape=[batch_size, 1]))
                # one_label = tf.cond(p, tf.constant([1, 0], tf.constant([0, 1])))
                one_loss = compute_loss(all_logits[i], true_label)
                all_loss.append(one_loss)

            total_loss = tf.constant(0.0)
            for loss in all_loss:
                total_loss += loss

            # start_positions = features["start_positions"]
            # end_positions = features["end_positions"]
            # emotion_label = features["emotion_label"]

            # end_loss = compute_loss(end_logits, end_positions)

            # total_loss = (0.5 * start_loss + 0.5 * end_loss )

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, True)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                export_outputs=None,
                training_chief_hooks=None,
                # training_hooks=[
                # evalute_hook(handle=handle,feed_handle=test_handle, run_op=total_loss, evl_step=10),
                # train_hook(handle,train_handle)
                # ],
                scaffold=scaffold_fn
            )
        # elif mode == tf.estimator.ModeKeys.EVAL:
        #     seq_length = modeling.get_shape_list(input_ids)[1]
        #
        #     def compute_loss(logits, positions):
        #         one_hot_positions = tf.one_hot(
        #             positions, depth=seq_length, dtype=tf.float32)
        #         log_probs = tf.nn.log_softmax(logits, axis=-1)  # todo 为什么使用logsoftmax()???
        #         loss = -tf.reduce_mean(
        #             tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        #         return loss
        #
        #     def compute_emotion_loss(logits, labels):
        #         one_hot_labels = tf.one_hot(labels, depth=3, dtype=tf.float32)
        #         log_probs = tf.nn.log_softmax(logits, axis=-1)
        #         loss = - tf.reduce_mean(tf.reduce_sum(one_hot_labels * log_probs, axis=-1))
        #         return loss
        #
        #     start_positions = features["start_positions"]
        #     end_positions = features["end_positions"]
        #     emotion_label = features["emotion_label"]
        #
        #     start_loss = compute_loss(start_logits, start_positions)
        #     end_loss = compute_loss(end_logits, end_positions)
        #     emotion_loss = compute_emotion_loss(emotion_logits, emotion_label)
        #
        #     total_loss = (0.45 * start_loss + 0.45 * end_loss + 0.1 * emotion_loss)
        #     def metric_fn():
        #         pass
        #     output_spec = tf.estimator.EstimatorSpec(
        #         mode=mode,
        #         loss=total_loss,
        #         eval_metric_ops=None,
        #         scaffold=scaffold_fn,
        #         evaluation_hooks=None,
        #     )
        elif mode == tf.estimator.ModeKeys.PREDICT:
            example_id = features["example_id"]
            predictions = {
                "unique_ids": unique_ids,
                "example_id":example_id,
                # "res":res_probility,
                # "start_logits": start_logits,
                # "end_logits": end_logits,
                # "predict": logits
                # "emotion": emotion_logits
            }
            for index,logits in enumerate(all_logits):
                predictions["res{}".format(index)]=tf.nn.softmax(logits, axis=-1)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions, scaffold=scaffold_fn)
        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder, batch, answer_len, all_max):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "answer_mark": tf.FixedLenFeature([answer_len], tf.int64),
        "content_mark": tf.FixedLenFeature([all_max - answer_len - 2], tf.int64),
        "example_id":tf.FixedLenFeature([],tf.string),
        "label_mark":tf.FixedLenFeature([answer_len * 20], tf.float32)
    }

    if is_training:
        # name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        # name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
        # name_to_features["emotion_label"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["label"] = tf.FixedLenFeature([40], tf.float32)


    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params[batch]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return d

    return input_fn


def my_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    if is_training:
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["emotion_label"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(batch):
        """The actual input function."""
        batch_size = batch

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return d

    return input_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", ])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file):
    """Write final predictions to the json file and log-odds of null if needed."""
    tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
    tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    diff_bath = 0
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        my_emotion = None
        for (feature_index, feature) in enumerate(features):
            try:
                result = unique_id_to_result[feature.unique_id]
                # my_emotion = result.emotion_logits
            except Exception as e:
                diff_bath += 1
                print("最后一个bath不够，丢了，导致result和feature不等长，出现用featureid查resultid的out of index")
                print("他们相差了:{}个".format(diff_bath))
                continue
            #     print("实在不行吧batchsize设置为1吧。哈哈哈")
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if FLAGS.version_2_with_negative:  # 确定没有回答的概率
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= (len(feature.tokens) - max_answer_length - 2):
                        continue
                    if end_index >= (len(feature.tokens) - max_answer_length - 2):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        if FLAGS.version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            # try:
            feature = features[pred.feature_index]
            # except Exception as e:
            #     print("have error!!")
            #     print("error:{},{}".format(e, example_index))
            if pred.start_index >= 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # if we didn't inlude the empty option in the n-best, inlcude it
        if FLAGS.version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="", start_logit=null_start_logit,
                        end_logit=null_end_logit))
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)
        # emotion_probs = _compute_softmax(my_emotion)
        # num2emotion = {0: "POS", 1: "NORM", 2: "NEG"}
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            # output["emotion_pro"] = emotion_probs
            # if output["text"] == "empty":
            #     output["emotion_text"] = ""
            # else:
            #     output["emotion_text"] = num2emotion[np.array(emotion_probs).argmax()]
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not FLAGS.version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold

            if not best_non_null_entry:
                score_diff = FLAGS.null_score_diff_threshold + 10000000000
            else:
                score_diff = score_null - best_non_null_entry.start_logit - (
                    best_non_null_entry.end_logit)

            scores_diff_json[example.qas_id] = score_diff
            if score_diff > FLAGS.null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with tf.gfile.GFile(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with tf.gfile.GFile(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if FLAGS.version_2_with_negative:
        with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if FLAGS.verbose_logging:
            tf.logging.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if FLAGS.verbose_logging:
            tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["answer_mark"] = create_int_feature(feature.answer_mark)
        features["content_mark"] = create_int_feature(feature.content_mark)
        features["example_id"] = tf.train.Feature(bytes_list = tf.train.BytesList(value=[feature.example_id.encode(encoding="utf-8")]))
        features["label_mark"] = tf.train.Feature(
            float_list=tf.train.FloatList(value=feature.label_mark.flatten().tolist()))

        if self.is_training:
            # features["start_positions"] = create_int_feature([feature.start_position])
            # features["end_positions"] = create_int_feature([feature.end_position])
            # features["emotion_label"] = create_int_feature([feature.emotion_label])
            features["label"] = tf.train.Feature(float_list=tf.train.FloatList(value=feature.label))
            impossible = 0
            if feature.is_impossible:
                impossible = 1
            features["is_impossible"] = create_int_feature([impossible])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def validate_flags_or_throw(bert_config):
    """Validate the input FLAGS or throw an exception."""
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if FLAGS.do_train:
        if not FLAGS.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if FLAGS.do_predict:
        if not FLAGS.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
        raise ValueError(
            "The max_seq_length (%d) must be greater than max_query_length "
            "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    validate_flags_or_throw(bert_config)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    # is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    #     run_config = tf.contrib.tpu.RunConfig(
    #         cluster=tpu_cluster_resolver,
    #         master=FLAGS.master,
    #         model_dir=FLAGS.output_dir,
    #         save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    #         tpu_config=tf.contrib.tpu.TPUConfig(
    #             iterations_per_loop=FLAGS.iterations_per_loop,
    #             num_shards=FLAGS.num_tpu_cores,
    #             per_host_input_for_training=is_per_host))
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        tf_random_seed=None,
        save_summary_steps=10,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        session_config=None,
        keep_checkpoint_max=5,
        log_step_count_steps=100,
        train_distribute=None,
        device_fn=None

    )

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    train_writer_filename = None
    if FLAGS.do_train:
        if FLAGS.reload_data:
            train_examples = read_squad_examples(
                input_file=FLAGS.train_file, is_training=True)
            num_train_steps = int(
                len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
            num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

            # Pre-shuffle the input to avoid having to make a very large shuffle
            # buffer in in the `input_fn`.
            rng = random.Random(12345)
            rng.shuffle(train_examples)
            # We write to a temporary file to avoid storing very large constant tensors
            # in memory.
            # train_writer = FeatureWriter(
            #     filename=os.path.join(FLAGS.output_dir,
            #                           "train_{}_{}_end.tf_record".format(num_train_steps, num_warmup_steps)),
            #     is_training=True)
            # num_train_steps=10000
            # num_warmup_steps = 10
            train_writer = FeatureWriter(
                filename=os.path.join(FLAGS.output_dir,
                                      "train_{}_{}.tf_record".format(num_train_steps, num_warmup_steps)),
                is_training=True)
            train_writer_filename = train_writer.filename
            convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=FLAGS.max_seq_length,
                doc_stride=FLAGS.doc_stride,
                max_query_length=FLAGS.max_query_length,
                is_training=True,
                output_fn=train_writer.process_feature)
            train_writer.close()

            tf.logging.info("***** Running training *****")
            tf.logging.info("  Num orig examples = %d", len(train_examples))
            tf.logging.info("  Num split examples = %d", train_writer.num_features)
            tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
            tf.logging.info("  Num steps = %d", num_train_steps)
            del train_examples
        else:
            tfrecord_list = os.listdir("res")
            train_file = None
            eval_file = None
            predict_file = None
            for name in tfrecord_list:
                if name.find("train") != -1:
                    train_file = name
                elif name.find("eval") != -1:
                    eval_file = name
                elif name.find("predict") != -1:
                    predict_file = name
            train_file = "train_33466_3346_.tf_record"
            num_train_steps = int(train_file.split("_")[1])
            num_warmup_steps = int(train_file.split("_")[2])
            train_writer_filename = os.path.join(FLAGS.output_dir, name)
            train_writer_filename = "D:\\train_33466_3346_.tf_record"
    # train_iter = my_input_fn_builder("res/train_124728_12472_end.tf_record",FLAGS.max_seq_length,True,True)(FLAGS.train_batch_size)
    # test_iter = my_input_fn_builder("res/dev.tf_record",FLAGS.max_seq_length,True,True)(32)
    # train_iterator = train_iter.make_initializable_iterator()
    # test_iterator = test_iter.make_initializable_iterator()
    # train_handle = tf.Session().run(train_iterator.string_handle())
    # test_handle = tf.Session().run(test_iterator.string_handle())
    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        answer_max=FLAGS.max_query_length,
        all_max=FLAGS.max_seq_length)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    # estimator = tf.contrib.tpu.TPUEstimator(
    #     use_tpu=FLAGS.use_tpu,
    #     model_fn=model_fn,
    #     config=run_config,
    #     train_batch_size=FLAGS.train_batch_size,
    #     predict_batch_size=FLAGS.predict_batch_size)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"train_batch_size": FLAGS.train_batch_size,
                "eval_batch_size": 128,
                "predict_batch_size": FLAGS.predict_batch_size},  # params可以传给mofel_fn和input_fn
        warm_start_from=None,
    )

    if FLAGS.do_train:
        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.
        # train_writer = FeatureWriter(
        #     filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
        #     is_training=True)
        # convert_examples_to_features(
        #     examples=train_examples,
        #     tokenizer=tokenizer,
        #     max_seq_length=FLAGS.max_seq_length,
        #     doc_stride=FLAGS.doc_stride,
        #     max_query_length=FLAGS.max_query_length,
        #     is_training=True,
        #     output_fn=train_writer.process_feature)
        # train_writer.close()
        #
        # tf.logging.info("***** Running training *****")
        # tf.logging.info("  Num orig examples = %d", len(train_examples))
        # tf.logging.info("  Num split examples = %d", train_writer.num_features)
        # tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        # tf.logging.info("  Num steps = %d", num_train_steps)
        # del train_examples

        train_input_fn = input_fn_builder(
            input_file=train_writer_filename,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            batch="train_batch_size",
            answer_len=FLAGS.max_query_length,
            all_max=FLAGS.max_seq_length)
        debug_hook = tf_debug.LocalCLIDebugHook()
        debug_hook.add_tensor_filter("nan_and_inf", tf_debug.has_inf_or_nan)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps
                        # ,hooks=[debug_hook]
                        )
    # if True:
    #     # record_eval_arg = os.path.basename(eval_file)[-10].split("_")
    #
    #     # tf.logging.info("***** Running evaluation *****")
    #     # tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    #
    #     # This tells the estimator to run through the entire set.
    #     eval_steps = 100
    #     # However, if running eval on the TPU, you will need to specify the
    #     # number of steps.
    #
    #     eval_drop_remainder = True
    #     eval_input_fn = input_fn_builder(
    #         input_file="res/train_124728_12472_end.tf_record",
    #         seq_length=FLAGS.max_seq_length,
    #         is_training=True,
    #         drop_remainder=eval_drop_remainder,
    #         batch="eval_batch_size")
    #
    #     result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    #     # , hooks=[tf_debug.LocalCLIDebugHook(ui_type="readline")])
    #
    #     # output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    #     # with tf.gfile.GFile(output_eval_file, "w") as writer:
    #     tf.logging.info("***** Eval results *****")
    #     for key in sorted(result.keys()):
    #         tf.logging.info("  %s = %s", key, str(result[key]))
    #             # writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        eval_examples = read_squad_examples(
            input_file=FLAGS.predict_file, is_training=False)

        eval_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
            is_training=False)
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=False,
            output_fn=append_feature)
        eval_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        # all_results = []

        predict_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False,
            batch="predict_batch_size",
            answer_len=FLAGS.max_query_length,
            all_max=FLAGS.max_seq_length)

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        all_results = []
        wf = open("res/result.txt","w",encoding="utf-8")
        for mm,result in enumerate(estimator.predict(
                predict_input_fn, yield_single_examples=True
                # ,hooks=[tf_debug.LocalCLIDebugHook(ui_type="readline")]
        )):
            # if len(all_results) % 1000 == 0:
            tf.logging.info("Processing example: %d" % (mm))
            # wf.write(json.dumps(result)+"/n")
            example_id = result["example_id"].decode()
            res_lt = []
            for i in range(20):
                res_lt.append(result["res{}".format(i)].tolist())
            res = [ i[0] for index,i in enumerate(res_lt)]
            wf.write(example_id + "\t")
            wf.write(",".join([str(i)for i in res]))
            wf.write("\n")
        wf.close()
            # start_logits = [float(x) for x in result["start_logits"].flat]
            # end_logits = [float(x) for x in result["end_logits"].flat]
            # emotion_logits = [float(x) for x in result["emotion"].flat]
        #     all_results.append(
        #         RawResult(
        #             unique_id=unique_id,
        #             start_logits=start_logits,
        #             end_logits=end_logits
        #             # ,emotion_logits=emotion_logits
        #         ))
        #
        # output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
        # output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")
        # output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")
        #
        # write_predictions(eval_examples, eval_features, all_results,
        #                   FLAGS.n_best_size, FLAGS.max_answer_length,
        #                   FLAGS.do_lower_case, output_prediction_file,
        #                   output_nbest_file, output_null_log_odds_file)


if __name__ == "__main__":
    tf.app.run()

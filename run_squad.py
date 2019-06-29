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
import numpy as np
from lib.bert.myhook import evalute_hook,train_hook
from tensorflow.python import debug as tfdbg

flags = tf.flags

FLAGS = flags.FLAGS

run_config = json.load(open("config_file/run_squad_config.json","r",encoding="utf-8"))
## Required parameters
flags.DEFINE_string(
    "bert_config_file", run_config["bert_config_file"],
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", run_config["vocab_file"],
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    # "output_dir", "D:\\programing\\souhumodel\\BERT_core_match",
    "output_dir", run_config["output_dir"],
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", run_config["train_file"],
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", run_config["predict_file"],
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", run_config["init_checkpoint"],
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", run_config["do_lower_case"]=="True",
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_bool("reload_data", run_config["reload_data"]=="True", "是否重新加载数据并生成tfrecord文件")
flags.DEFINE_integer(
    "max_seq_length", run_config["max_seq_length"],
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", run_config["doc_stride"],
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", run_config["max_query_length"],
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", run_config["do_train"]=="True", "Whether to run training.")

flags.DEFINE_bool("do_predict", run_config["do_predict"]=="True", "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", run_config["train_batch_size"], "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", run_config["predict_batch_size"],
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", run_config["learning_rate"], "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", run_config["num_train_epochs"],
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", run_config["warmup_proportion"],
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", run_config["save_checkpoints_steps"],
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", run_config["iterations_per_loop"],
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", run_config["n_best_size"],
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", run_config["max_answer_length"],
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
                 is_impossible=False
                 ):
        self.emotion_label = emotion_label
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

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
                 unique_id,
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
                 is_impossible=None):
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


class sohudata():
    def __init__(self, data, istrain=True):
        paten = re.compile(r"\n+")
        self.newsid = data["newsId"]
        self.title = data['title']
        self.content = self.title + data['content'].strip() + self.title
        self.content = re.sub(paten," ",self.content)
        self.entity_mon = list()
        if istrain:
            for dic in data['coreEntityEmotions']:
                self.entity_mon.append((dic["entity"].strip(), dic["emotion"].strip()))


def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""

    # with tf.gfile.Open(input_file, "r") as reader:
    #     input_data = json.load(reader)["data"]

    def transform_data2examples(entry):
        def split_paragraph(entry):
            '''
            这是一个最大化利用监督数据的分割方法
            分割条件：不同实体之间包含一个语义分割符号（。，等）那么该符号即为分割点。
            :param paragraph:
            :return:
            '''

            def get_answer(context, cores):
                my_answer = []
                for core_entity in cores:
                    p = context.find(core_entity, 0)
                    while p != -1:
                        my_answer.append((p, core_entity))
                        p = context.find(core_entity, p + 1)
                return my_answer

            pattern = re.compile(r"\n+")
            context = re.sub(pattern, "", entry.content)
            cores = [i[0] for i in entry.entity_mon]
            max_len_core = sum([len(i) for i in cores])
            res_lt = sorted([i[0] for i in get_answer(context, cores)])
            # context = list(context)
            split_point = list()
            # split_point.append(0)
            for i in range(len(res_lt) - 1):
                find = context.find("。", res_lt[i], res_lt[i + 1])
                if find == -1:
                    find = context.find("，", res_lt[i], res_lt[i + 1])
                    if find == -1:
                        continue
                    else:
                        if find <= res_lt[i] + max_len_core:
                            continue
                        else:
                            split_point.append(find + 1)
                else:
                    split_point.append(find + 1)
            # split_point.append(len(context)-1)
            if len(split_point) == 0:
                return [context]
            contexts_indix = list()
            for i in range(len(split_point) - 1):
                contexts_indix.append((split_point[i], split_point[i + 1]))
            contexts = list()
            try:
                contexts.append(context[:split_point[0]])
            except Exception as e:
                print("最大化利用监督问题：{}".format(e))
            for i in contexts_indix:
                contexts.append(context[i[0]:i[1]])
            contexts.append(context[split_point[-1]:])
            return contexts

        def get_all_answer(contexts, ansowers):
            res = list()
            for context in contexts:
                context = " ".join(list(context))
                res_context = []
                for core_entity, core_emotion in ansowers:
                    core_entity = " ".join(list(core_entity))
                    p = context.find(core_entity, 0)
                    while p != -1:
                        res_context.append((p, core_entity, core_emotion))
                        p = context.find(core_entity, p + 1)
                res.append(res_context)
                if len(res[0])==0:
                    print("wocao")
            return res

        def select_answer(all_res):
            def get_core_dic():
                dic_list = []
                for res in all_res:
                    core_dic = collections.defaultdict(list)
                    for i in res:
                        core_dic[i[1]].append((i[0],i[2]))
                    dic_list.append(core_dic)
                return dic_list
            all_res = get_core_dic()
            is_imp = list()
            right_res = list()
            for res in all_res:
                if not res:
                    is_imp.append(True)
                    right_res.append((-1, "没有答案", "没得感情"))
                else:
                    is_imp.append(False)
                    # keys = res.keys()
                    keys = []
                    for i in res.keys():
                        keys.append(i)
                    key = keys[random.randint(0,len(keys)-1)]
                    dic_res = res[key][random.randint(0,len(res[key])-1)]
                    right_res.append((dic_res[0],key,dic_res[1]))  # 产生唯一解
            return is_imp, right_res


        # start = re.compile(r"^\n\n*")
        title = " ".join(list(entry.title))
        ansowers = entry.entity_mon
        # core_rate = None
        if is_training:
            contexts = split_paragraph(entry)
            res = get_all_answer(contexts, ansowers)
            is_imp, right_res = select_answer(res)
            # try:
            #     core_rate = float(len(res))/float(sum([len(i) for i in res]))
            # except Exception as e:
            #     print(e)
            # if core_rate < 0.15:
            #     print("laji")
            for text in is_imp:
                if text:
                    print("卧槽，有一些段落没有实体，请调bug吧，哈哈！！！")
                    return -1
        else:
            pat = re.compile(r"，|。")
            contexts = list()
            temp_context = ""
            for context in re.split(pat,entry.content):
                temp_context += context
                if len(temp_context) < 60:
                    continue
                else:
                    contexts.append(" ".join(temp_context))
                    temp_context = ""
            if len(temp_context) != 0:
                contexts.append(" ".join(temp_context))
            right_res = list()
            for _ in range(len(contexts)):
                right_res.append([None, None, None])

        # print("问题数据数：{}".format(i))
        dic_enty = {"title": title, "paragraphs": [{"context": " ".join(list(context)), "qas": [
            {"answers": [{"answer_start": right_res[para_index][0], "text": right_res[para_index][1],
                          "emotion_label": right_res[para_index][-1]}],
             "question": title, "id": entry.newsid + str(para_index), "is_impossible": False}]} for
                                                   para_index, context in
                                                   enumerate(contexts)]}
        return dic_enty

    input_data = list()
    # core_rate = list()
    with open(input_file, "r", encoding="utf-8") as rf:
        i = 0
        for indx, data in enumerate(rf):
            shu_data = sohudata(json.loads(data), is_training)
            if len(shu_data.entity_mon) == 0 and is_training:
                i = i + 1
                print("shouhudata,没有实体")
                continue
            one_input_data = transform_data2examples(shu_data)
            # core_rate.append(temp_rate)
            if one_input_data == -1:
                i = i + 1
                continue
            input_data.append(one_input_data)
            # print(temp_rate)
            if run_config["debug"] == 1:
                if indx == 20: break
        # print(core_rate)
        # print("监督利用率：{}".format(sum(core_rate)/len(core_rate)))
        print("问题数据数：{}".format(i))

    # test = transform_data2examples(input_data[2])

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)
            emotion2num = {"POS": 0, "NORM": 1, "NEG": 2, "没得感情": 1}
            for qa in paragraph["qas"]:
                emotion_label = None
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False

                if is_training:
                    emotion_label = emotion2num[qa["answers"][0]["emotion_label"]]
                    if FLAGS.version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length -
                                                           1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            tokenization.whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                               actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    emotion_label=emotion_label,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible
                )
                examples.append(example)

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
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
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            emotion = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            if example_index < 20:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
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
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    tf.logging.info("start_position: %d" % (start_position))
                    tf.logging.info("end_position: %d" % (end_position))
                    tf.logging.info(
                        "answer: %s" % (tokenization.printable_text(answer_text)))
            if is_training:
                emotion = example.emotion_label

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                emotion_label=emotion,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible)

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


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()


    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
        "cls/squad/output_weights", [2, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden,
                                     [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
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

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (start_logits, end_logits) = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

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
                one_hot_positions = tf.one_hot(
                    positions, depth=seq_length, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)  # todo 为什么使用logsoftmax()???
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                return loss



            start_positions = features["start_positions"]
            end_positions = features["end_positions"]
            emotion_label = features["emotion_label"]

            start_loss = compute_loss(start_logits, start_positions)
            end_loss = compute_loss(end_logits, end_positions)


            total_loss = (0.5 * start_loss + 0.5 * end_loss )

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, run_config["freeze"])

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
            predictions = {
                "unique_ids": unique_ids,
                "start_logits": start_logits,
                "end_logits": end_logits,
                # "emotion": emotion_logits
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions, scaffold=scaffold_fn)
        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder, batch):
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
                                   ["unique_id", "start_logits", "end_logits",])


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
                print("实在不行吧batchsize设置为1吧。哈哈哈")
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
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
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
            try:
                feature = features[pred.feature_index]
            except Exception as e:
                print("have error!!")
                print("error:{},{}".format(e, example_index))
            if pred.start_index > 0:  # this is a non-null prediction
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
    # test_wf = open(output_nbest_file, "w", encoding="unicode")
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

        if self.is_training:
            features["start_positions"] = create_int_feature([feature.start_position])
            features["end_positions"] = create_int_feature([feature.end_position])
            features["emotion_label"] = create_int_feature([feature.emotion_label])
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
    tf_run_config = tf.estimator.RunConfig(
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
            train_writer = FeatureWriter(
                filename=os.path.join(FLAGS.output_dir,
                                      "train_{}_{}_.tf_record".format(num_train_steps,num_warmup_steps)),
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
            tfrecord_list = os.listdir(FLAGS.output_dir)
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
            # train_file = "train_33466_3346_.tf_record"
            num_train_steps = int(train_file.split("_")[1])
            num_warmup_steps = int(train_file.split("_")[2])
            train_writer_filename = os.path.join(FLAGS.output_dir, train_file)
            # train_writer_filename = "D:\\train_33466_3346_.tf_record"
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
        use_one_hot_embeddings=FLAGS.use_tpu)

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
        config=tf_run_config,
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
            batch="train_batch_size")
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps,hooks=[tfdbg.TensorBoardDebugHook(grpc_debug_server_addresses="localhost:11111")])
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
            batch="predict_batch_size")

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        all_results = []
        for result in estimator.predict(
                predict_input_fn, yield_single_examples=True):
            if len(all_results) % 1000 == 0:
                tf.logging.info("Processing example: %d" % (len(all_results)))
            unique_id = int(result["unique_ids"])
            start_logits = [float(x) for x in result["start_logits"].flat]
            end_logits = [float(x) for x in result["end_logits"].flat]
            # emotion_logits = [float(x) for x in result["emotion"].flat]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits
                    # ,emotion_logits=emotion_logits
                ))

        output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
        output_nbest_file = run_config["nbest_name"]
        output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")

        write_predictions(eval_examples, eval_features, all_results,
                          FLAGS.n_best_size, FLAGS.max_answer_length,
                          FLAGS.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file)


if __name__ == "__main__":
    tf.app.run()

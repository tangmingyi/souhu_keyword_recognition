import tensorflow as tf
import collections
import json
import numpy as np
import math
import re
from lib.bert.modeling import gelu
import lib.bert.optimization


class InputFeatures(object):  # 输入的数据转化成为的特征（数字化）
    """A single set of features of data."""

    def __init__(self,
                 content_vactor,
                 answer_vactor,
                 label_ids,
                 new_id,
                 content_mark,
                 answer_mark):
        self.content_vactor = content_vactor
        self.answer_vactor = answer_vactor
        self.label_id = label_ids
        self.new_id = new_id
        self.content_mark = content_mark
        self.answer_mark = answer_mark


def convert_single_example(example, sequence_max_len):
    features = []
    for info in example["info"]:
        for k, v in info.items():
            new_id = k
            content_vactor = np.array([i["layers"][0]["values"] for i in v["content"]])
            content_mark = None
            # if content_vactor.shape[0] == sequence_max_len:
            #     print("len == 128")
            if content_vactor.shape[0] >= sequence_max_len:
                # print("debug128")
                if content_vactor.shape[0] > sequence_max_len:
                    content_vactor = content_vactor[:sequence_max_len, :]
                content_mark = np.ones((sequence_max_len))
            else:
                # print("debug****")
                temp = np.zeros(shape=(sequence_max_len - content_vactor.shape[0], content_vactor.shape[1]))
                temp_mark = np.zeros(shape=(sequence_max_len - content_vactor.shape[0]))
                dim_content_vactor = content_vactor.shape[0]
                content_vactor = np.concatenate((content_vactor, temp), axis=0)
                content_mark = np.append(np.ones(dim_content_vactor), temp_mark)
                # if content_mark.shape[0] != sequence_max_len or content_vactor.shape[0] != sequence_max_len:
                #     print("debug")
            answer_vactor = np.array([i["layers"][0]["values"] for i in v["title"]])
            answer_mark = None
            if answer_vactor.shape[0] >= sequence_max_len:
                if answer_vactor.shape[0] > sequence_max_len:
                    answer_vactor = answer_vactor[:sequence_max_len, :]
                content_mark = np.ones((sequence_max_len))
            else:
                temp = np.zeros(shape=(sequence_max_len - answer_vactor.shape[0], answer_vactor.shape[1]))
                temp_mark = np.zeros(shape=(sequence_max_len - answer_vactor.shape[0]))
                dim_answer_vactor = answer_vactor.shape[0]
                answer_vactor = np.concatenate((answer_vactor, temp), axis=0)
                answer_mark = np.concatenate((np.ones(dim_answer_vactor), temp_mark), axis=0)
                # if answer_mark.shape[0] != sequence_max_len or answer_vactor.shape[0]!=sequence_max_len:
                #     print("debug answer")
            label_id = v["label"]
            feature = InputFeatures(content_vactor, answer_vactor, label_id, new_id, content_mark, answer_mark)
            features.append(feature)
    return features


def file_based_convert_examples_to_features(data_file, output_file, sequence_max_len, is_train=True):
    """Convert a set of `InputExample`s to a TFRecord file."""
    writer = tf.python_io.TFRecordWriter(output_file)
    rf = open(data_file, "r", encoding="utf-8")
    for (ex_index, line) in enumerate(rf):
        if ex_index % 10 == 0:
            tf.logging.info("Writing example %d" % (ex_index))
        example = json.loads(line.strip())
        many_feature = convert_single_example(example, sequence_max_len)

        def create_int_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        for feature in many_feature:
            features = collections.OrderedDict()  # tensorflow feature字典
            features["new_id"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[feature.new_id.encode(encoding="utf-8")]))
            features["content_vactor"] = create_int_feature(feature.content_vactor.ravel())
            features["answer_vactor"] = create_int_feature(feature.answer_vactor.ravel())
            features["content_mark"] = create_int_feature(feature.content_mark)
            features["answer_mark"] = create_int_feature(feature.answer_mark)

            # features["segment_ids"] = create_int_feature(feature.segment_ids)
            if is_train:
                features["label_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature.label_id)))

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, is_training,
                                drop_remainder, bert_hidden_size, max_sequence_len):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "new_id": tf.FixedLenFeature([], tf.string),
        "content_vactor": tf.FixedLenFeature([bert_hidden_size * max_sequence_len], tf.float32),
        "answer_vactor": tf.FixedLenFeature([bert_hidden_size * max_sequence_len], tf.float32),
        "content_mark": tf.FixedLenFeature([max_sequence_len], tf.float32),
        "answer_mark": tf.FixedLenFeature([max_sequence_len], tf.float32),
        # "label_ids": tf.FixedLenSequenceFeature([2],tf.int64),
        # "is_real_example": tf.FixedLenFeature(tf.int64),
    }
    if is_training:
        name_to_features["label_ids"] = tf.FixedLenFeature([2], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)  # record是example的序列化，通过这个函数解析为features字典

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        # for name in list(example.keys()):
        #     t = example[name]
        #     if t.dtype == tf.int64:
        #         t = tf.to_int32(t)
        #     example[name] = t

        return example

    def input_fn(batch_size):
        """The actual input function."""
        batch_size = batch_size

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)  # 建立dataset数据的来源
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                # 对每个元素应用map函数进行张量的预处理；dataset可能会将读取的record原始序列张量，传入其中
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)  # 返回所有init_checkpoint中的variables

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    attention_ffc = None):
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    def reshape_to_matrix(input_tensor):
        """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
        ndims = input_tensor.shape.ndims
        if ndims < 2:
            raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                             (input_tensor.shape))
        if ndims == 2:
            return input_tensor

        width = input_tensor.shape[-1]
        output_tensor = tf.reshape(input_tensor, [-1, width])
        return output_tensor

    def dropout(input_tensor, dropout_prob):
        if dropout_prob is None or dropout_prob == 0.0:
            return input_tensor
        output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
        return output

    if batch_size == None:
        batch_size = tf.shape(from_tensor)[0]
    from_seq_length = from_tensor.shape[1].value
    to_seq_length = to_tensor.shape[1].value

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_weight = attention_ffc["query_weight"]
    query_bias = attention_ffc["query_bias"]
    query_layer = tf.nn.bias_add(tf.matmul(from_tensor_2d,query_weight,transpose_b=True),query_bias)
    # query_layer = tf.layers.dense(
    #     from_tensor_2d,
    #     num_attention_heads * size_per_head,
    #     activation=query_act,
    #     name="query",
    #     kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))

    # `key_layer` = [B*T, N*H]
    key_weight = attention_ffc["key_weight"]
    key_bias = attention_ffc["key_bias"]
    key_layer = tf.nn.bias_add(tf.matmul(to_tensor_2d,key_weight,transpose_b=True),key_bias)

    # key_layer = tf.layers.dense(
    #     to_tensor_2d,
    #     num_attention_heads * size_per_head,
    #     activation=key_act,
    #     name="key",
    #     kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))

    # `value_layer` = [B*T, N*H]
    value_weight = attention_ffc["value_weight"]
    value_bias = attention_ffc["value_bias"]
    value_layer = tf.nn.bias_add(tf.matmul(to_tensor_2d,value_weight,transpose_b=True),value_bias)
    # value_layer = tf.layers.dense(
    #     to_tensor_2d,
    #     num_attention_heads * size_per_head,
    #     activation=value_act,
    #     name="value",
    #     kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


class attention_model():

    def __init__(self, content_vactor, answer_vactor,
                 attention_mark, bert_hidden_size=768,
                 num_attention_heads=12, ffc_wight=None):
        with tf.variable_scope("match_lay"):
            with tf.variable_scope("attention"):
                self.MP = attention_layer(from_tensor=content_vactor,
                                          to_tensor=answer_vactor,
                                          num_attention_heads=num_attention_heads,
                                          size_per_head=int(bert_hidden_size / num_attention_heads),
                                          attention_probs_dropout_prob=0.1,
                                          initializer_range=0.02,
                                          do_return_2d_tensor=False,
                                          attention_mask=attention_mark,
                                          attention_ffc = ffc_wight["attention"])
            with tf.variable_scope("representation"):
                self.HP = content_vactor
                temp_vactor = tf.concat([tf.subtract(self.MP, self.HP), tf.multiply(self.MP, self.HP)], axis=-1)
                batch_size = tf.shape(temp_vactor)[0]
                seq_len = temp_vactor.shape[1].value
                temp_vactor = tf.reshape(temp_vactor,shape=[-1,bert_hidden_size*2])
                repersentation_weight = ffc_wight["reper_weight"]
                repersentation_bias = ffc_wight["reper_bias"]
                self.repersentation = tf.reshape(tf.nn.relu(
                    tf.nn.bias_add(tf.matmul(temp_vactor, repersentation_weight, transpose_b=True),
                                   repersentation_bias)),shape=[batch_size,seq_len,bert_hidden_size])
                # self.repersentation = tf.layers.dense(temp_vactor, bert_hidden_size, activation=tf.nn.relu,
                #                                       use_bias=True,
                #                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def get_repersentation_vator(self):
        return self.repersentation

        # with tf.name_scope("loss"):
        #     final_hidden_shape = repersentation.shape
        #     batch_size = final_hidden_shape[0]
        #     seq_length = final_hidden_shape[1]
        #     hidden_size = final_hidden_shape[2]
        #
        #     output_weights = tf.get_variable(
        #         "cls/squad/output_weights", [2, hidden_size],
        #         initializer=tf.truncated_normal_initializer(stddev=0.02))
        #
        #     output_bias = tf.get_variable(
        #         "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())
        #
        #     final_hidden_matrix = tf.reshape(repersentation,
        #                                      [batch_size * seq_length, hidden_size])
        #     logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
        #     logits = tf.nn.bias_add(logits, output_bias)
        #
        #     logits = tf.reshape(logits, [batch_size, seq_length, 2])
        #     logits = tf.transpose(logits, [2, 0, 1])
        #
        #     unstacked_logits = tf.unstack(logits, axis=0)
        #     (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])
        #
        #     return (start_logits, end_logits)

# def build_model(features, is_train, learning_rate, num_train_steps, bert_hidden_size,
#                 init_checkpoint=None):
#     def get_attention_mark(content_mark, answer_mark):
#         pass
#
#     tf.logging.info("*** Features ***")
#     for name in sorted(features.keys()):
#         tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
#
#     new_id = features["new_id"]
#     batch_size = tf.shape(features["content_vactor"])[0]
#     # sequence_len = tf.divide(tf.shape(features["content_vactor"]), tf.constant(bert_hidden_size))
#     content_vactor = tf.reshape(features["content_vactor"], shape=[batch_size, -1, bert_hidden_size])
#     answer_vactor = tf.reshape(features["answer_vactor"], shape=[batch_size, -1, bert_hidden_size])
#     content_mark = features["content_mark"]
#     answer_mark = features["answer_mark"]
#     attention_mark = tf.multiply(tf.expand_dims(content_mark,axis=[2]), tf.expand_dims(answer_mark,axis=[1]))
#     label_ids = None
#     if is_train:
#         label_ids = features["label_ids"]
#         tf.logging.info("label_ids:{}".format(label_ids))
#     (start_logits, end_logits) = create_model(content_vactor, answer_vactor, attention_mark)
#     tvars = tf.trainable_variables()
#     initialized_variable_names = {}
#     scaffold_fn = None
#     if init_checkpoint:
#         (assignment_map, initialized_variable_names
#          ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
#         tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
#     tf.logging.info("**** Trainable Variables ****")
#     for var in tvars:
#         init_string = ""
#         if var.name in initialized_variable_names:
#             init_string = ", *INIT_FROM_CKPT*"
#         tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
#                         init_string)
#
#     def compute_loss(logits, positions, depth):
#         one_hot_positions = tf.one_hot(
#             positions, depth=depth, dtype=tf.float32)
#         log_probs = tf.nn.log_softmax(logits, axis=-1)
#         loss = -tf.reduce_mean(
#             tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
#         return loss
#
#     start_positions = label_ids[0]
#     end_positions = label_ids[-1]
#
#     start_loss = compute_loss(start_logits, start_positions, start_logits.shape[-1].value)
#     end_loss = compute_loss(end_logits, end_positions, start_logits.shape[-1].value)
#
#     total_loss = (0.5 * start_loss + 0.5 * end_loss)
#     train_op = optimization.create_optimizer(
#         total_loss, learning_rate, num_train_steps, int(0.1 * num_train_steps), False, False)
#     return train_op, total_loss,new_id
#
#
# def main():
#     LEARNING_RATE = 5e-5
#     NUM_TRAIN_STEPS = 30000
#     BERT_HIDDEN_SIZE = 768
#     MAX_SEQ_LEN = 128
#     tf.logging.set_verbosity(tf.logging.INFO)
#     # file_based_convert_examples_to_features("data/merge_token.json", "data/train.tf_record", MAX_SEQ_LEN)
#     input_fn = file_based_input_fn_builder("data/train.tf_record", True, True, BERT_HIDDEN_SIZE, MAX_SEQ_LEN)
#     data_set = input_fn(2)
#     iter = data_set.make_one_shot_iterator()
#     features = iter.get_next()
#     train_op, total_loss ,new_id= build_model(features, True, LEARNING_RATE, NUM_TRAIN_STEPS, BERT_HIDDEN_SIZE,init_checkpoint="model/test.ckpt")
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         for i in range(NUM_TRAIN_STEPS):
#             if i % 100==0:
#                 saver.save(sess,"./model/test.ckpt",global_step=i)
#             # tf.logging.info("begin_run")
#             _, loss,myid = sess.run([train_op, total_loss,new_id])
#             tf.logging.info("id:{} :{}".format(myid,loss))
#
#
#
# if __name__ == '__main__':
#     main()

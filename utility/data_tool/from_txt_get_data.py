import re
import json
import numpy as np
import os
import collections


class souhu_data():
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



class sub_res_data():
    def __init__(self, line, istrain, max_query_length):
        temp = line.strip().split("\t")
        self.new_id = temp[0]
        self.predict_res = temp[1].split("*|||*")
        try:
            self.res_index = [[int(j) for j in i.split(",")] for i in temp[4].split("*|||*")]
        except Exception as e:
            print(e)
        self.len_paragraph = int(temp[5].split("*|||*")[0])
        self.score = [float(i) for i in temp[3].split("*|||*")]
        self.vote_score = temp[2].split("*|||*")
        if istrain:
            self.res = temp[6].split("*|||*")
            self.res_point, self.tag = self._get_res_point()
            self.res_point = self._get_one_hot_labels()
        self.res_mark = self._get_predict_label_mark(max_query_length)

    def _get_res_point(self):
        res_lt = []
        tag = "good_data"
        for res in self.res:
            try:
                res_lt.append(self.predict_res.index(res))
            except ValueError as e:
                continue
        if len(res_lt) == 0:
            tag = "bad_data"
        elif max(res_lt) > 450: #todo:这里写的不对，后期改为检测文本长度
            tag = "bad_data"
        return res_lt, tag

    def _add_right_res(self):
        pass

    def _get_one_hot_labels(self):
        label = np.zeros(shape=(40))
        for i in range(20):
            if i in self.res_point:
                label[2 * i] = 1
            else:
                label[2 * i + 1] = 1
        return label

    def _get_predict_label_mark(self, max_query_length):
        label = np.zeros(shape=(20, max_query_length))
        end = -1
        for i, res_i in enumerate(self.predict_res):
            start = end + 1  # end 指向@符号
            end = start + len(res_i)
            if end > max_query_length:
                break
            for j in range(start, end):
                label[i][j] = 1
        return label


class sub_result():
    def __init__(self, new_id, res_prediction):
        self.new_id = new_id
        self.res_prediction = res_prediction


class train_data():
    def __init__(self, new_id, content, predict, label=None, label_mark=None):
        self.new_id = new_id
        self.content = content
        self.predict = predict
        self.label = label
        self.label_mark = label_mark


class data_util():
    def __init__(self):
        self.souhu_data = {}
        self.sub_data = []
        self.train_data = []

    def get_souhu_data(self, file_name, istrain):
        for line in open(file_name, "r", encoding="utf-8"):
            data = souhu_data(json.loads(line.strip()), istrain)
            self.souhu_data[data.newsid] = data

    def get_sub_res_data(self, file_name, istrain, max_query_length):
        for line in open(file_name, "r", encoding="utf-8"):
            data = sub_res_data(line.strip(), istrain, max_query_length)
            if istrain:
                if data.tag == "good_data":
                    self.sub_data.append(data)
            else:
                self.sub_data.append(data)

    def get_sub_train_data(self):
        assert len(self.souhu_data) != 0
        assert len(self.sub_data) != 0
        for sub_data in self.sub_data:
            self.train_data.append(
                train_data(sub_data.new_id, self.souhu_data[sub_data.new_id].content, sub_data.predict_res,
                           sub_data.res_point, sub_data.res_mark))

    def get_sub_prediction_data(self):
        assert len(self.souhu_data) != 0
        assert len(self.sub_data) != 0
        for sub_data in self.sub_data:
            self.train_data.append(
                train_data(sub_data.new_id, self.souhu_data[sub_data.new_id].content, sub_data.predict_res,
                           label_mark=sub_data.res_mark))




class Tool():
    @staticmethod
    def merge_txt(file_base, output_file):
        wf = open(output_file, "w", encoding="utf-8")
        file_names = os.listdir(file_base)
        for file_name in file_names:
            file_name = os.path.join(file_base, file_name)
            with open(file_name, "r", encoding="utf-8") as rf:
                for line in rf:
                    if line == "\n": continue
                    wf.write(line)
        wf.close()

    @staticmethod
    def split_txt(input_file, output_file_base, len_file):
        if not os.path.exists(output_file_base):
            os.mkdir(output_file_base)
        input_file_name = os.path.basename(input_file)
        with open(input_file, "r", encoding="utf-8") as rf:
            i = 1
            for index, line in enumerate(rf):
                # print(index)
                if index % len_file == 0:
                    if i != 1:
                        wf.close()
                    wf = open("{}{}.txt".format(os.path.join(output_file_base, input_file_name.split(".")[0]), i), "w",
                              encoding="utf-8")
                    i += 1
                wf.write(line)
            wf.close()


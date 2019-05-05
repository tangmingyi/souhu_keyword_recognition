import json
import re
import collections
from tqdm import tqdm
from utility.data_tool import from_txt_get_data
import os
import numpy as np

class PostProcessTool():
    @staticmethod
    def _get_merge_dic(raw_json_data):
        def clearn_data(data_list):
            value_list = list()
            punctuation = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+")
            for i in data_list:
                text = "".join(i["text"].split(" "))
                text = sorted(re.split(punctuation, text), key=lambda x: len(x), reverse=True)[0]
                i["text"] = text
                if len(text) < 2:
                    continue
                value_list.append(i)
            return value_list

        res = collections.defaultdict(list)
        all_res = {}
        print("begin merge data")
        for key, value in raw_json_data.items():
            value = clearn_data(value)
            value = [(i["text"], i["probability"]) for i in value]
            res[key[:8]].extend(value)
        for key, value in tqdm(res.items()):
            temp_res = collections.defaultdict(list)
            for k, v in value:
                temp_res[k].append(v)
            all_res[key] = temp_res
        return all_res

    @staticmethod
    def _get_sum_dic(all_res):
        temp_dic = {}
        print("begin sum")
        for key, value in tqdm(all_res.items()):
            inter_lt = []
            for k, v in value.items():
                if k == "empty": continue
                # pos = 0;
                # neg = 0;
                # norm = 0
                # for i in v:
                #     if i[1] == "POS":
                #         pos += 1
                #     elif i[1] == "NORM":
                #         norm += 1
                #     elif i[1] == "NEG":
                #         neg += 1
                # a_temp = [pos, neg, norm]
                # index = a_temp.index(max(a_temp))
                # num2emotion = {0: "POS", 1: "NORM", 2: "NEG"}
                inter_lt.append((k, sum(v)))
            aa = sorted(inter_lt, key=lambda x: x[1], reverse=True)
            temp_dic[key] = aa
        return temp_dic

    @staticmethod
    def _amend_res_with_title(res_dic, souhu_dic):
        """
        根据标题修正答案。
        :param res_dic:
        :param souhu_dic:
        :return:
        """
        def new_word(word, start_end_poit, title):
            new_word = title[start_end_poit[0]:start_end_poit[1] + 1]
            if word == new_word:
                return word, False
            for char in word:
                if char in new_word:
                    continue
                else:
                    return word, False
            if len(new_word) - len(word) <= 2:
                return new_word, True
            else:
                return word, False

        def maybe_word(start, end):
            words = list()
            for start_char in start:
                distants = []
                for end_char in end:
                    distant = end_char - start_char
                    if distant <= 0:
                        distant = 1000000
                    distants.append(distant)
                words.append((start_char, end[distants.index(min(distants))]))
            return words

        dic = collections.defaultdict(list)

        merte_num = 0
        print("begin fuse")
        change_num = 0
        for newid, cores in tqdm(res_dic.items()):
            temp_title = souhu_dic[newid].title
            for index, core in enumerate(cores[:5]):
                core_list = list(core[0])
                start_end = [core_list[0], core_list[-1]]
                start = []
                end = []
                p = temp_title.find(start_end[0], 0)
                while p != -1:
                    start.append(p)
                    p = temp_title.find(start_end[0], p + 1)

                p = temp_title.find(start_end[1], 0)
                while p != -1:
                    end.append(p)
                    p = temp_title.find(start_end[1], p + 1)
                if len(start) == 0 or len(end) == 0:
                    continue
                core_in_title = maybe_word(start, end)
                for core_sta_end in core_in_title:
                    word, change = new_word(core[0], core_sta_end, temp_title)
                    if not change:
                        continue
                    res_dic[newid][index] = (
                        word, res_dic[newid][index][1], res_dic[newid][index][2], res_dic[newid][index][3],
                        res_dic[newid][index][4], res_dic[newid][index][5])
                    change_num += 1
        print("change_num:{}".format(change_num))
        # dic[k] = sorted(dic[k], key=lambda x: x[1], reverse=True)
        # print("merge_number is :{}".format(merte_num))
        # return dic/

    @staticmethod
    def _merge_title(fuse_dic, souhu_dic):
        print("begin newid")
        merge_num = 0
        for newid, cores in tqdm(fuse_dic.items()):
            temp_title = souhu_dic[newid].title
            merge_temp = []
            for index, core in enumerate(cores[:10]):
                core_score = core[1]
                core = core[0]
                p = temp_title.find(core, 0)
                while p != -1:
                    merge_temp.append((p, p + len(core), core, core_score, index))
                    p = temp_title.find(core, p + 1)
            merge_temp = sorted(merge_temp, key=lambda x: x[3], reverse=True)
            for i in range(len(merge_temp) - 2):
                to_merge = merge_temp[i:i + 2]
                for one_to_merge in to_merge:
                    if one_to_merge[1] in [i[0] for i in to_merge]:
                        be_to_merge = to_merge[[i[0] for i in to_merge].index(one_to_merge[1])]
                        fuse_dic[newid][one_to_merge[-1]] = (
                            one_to_merge[2] + be_to_merge[2],
                            fuse_dic[newid][one_to_merge[-1]][1] + fuse_dic[newid][one_to_merge[-1]][1])
                        fuse_dic[newid][be_to_merge[-1]] = (be_to_merge[2], 0)
                        print(one_to_merge[2] + "\t" + be_to_merge[2])
                        # merge_temp.append((merge_a[0],merge_b[1],merge_a[2]+merge_b[2],merge_a[-1]))
                        merge_num += 1
            # merge_dic = {}
            # for dix,merge_a in enumerate(merge_temp):
            #     if fuse_dic[newid][merge_a[-1]][1] == 0:
            #         continue
            #     for merge_b in merge_temp:
            #         if fuse_dic[newid][merge_b[-1]][1] == 0:
            #             continue
            #         if merge_a[1]==merge_b[0]:
            #             fuse_dic[newid][merge_a[-1]] = (merge_a[2]+merge_b[2],fuse_dic[newid][merge_a[-1]][1]+fuse_dic[newid][merge_b[-1]][1])
            #             fuse_dic[newid][merge_b[-1]] = (merge_b[2],0)
            #             print(merge_a[2]+"\t"+merge_b[2])
            #             merge_temp.append((merge_a[0],merge_b[1],merge_a[2]+merge_b[2],merge_a[-1]))
            #             merge_num += 1
            fuse_dic[newid] = sorted(fuse_dic[newid], key=lambda x: x[1], reverse=True)
        print("merte num:{}".format(merge_num))

    # for i in range(2):
    @staticmethod
    def _sentense_togeter(raw_json):
        """
        按文章将句子聚在一起
        :param a:
        :return:
        """
        content_dic = collections.defaultdict(list)
        for paragraph_id, value in raw_json.items():
            content_dic[paragraph_id[:8]].append({paragraph_id: sorted([(i["text"], i["probability"]) for i in value],
                                                                       key=lambda x: x[1], reverse=True)[:]})
        return content_dic

    @staticmethod
    def _vote_and_sord(article_dic,numofget=20):
        res_dic = {}
        for new_id, value_lt in article_dic.items():
            inter_dic = collections.defaultdict(list)
            for para_num, paragraph in enumerate(value_lt):
                for k, v in paragraph.items():
                    for value in v:
                        inter_dic[value[0]].append((value[1], para_num + 1))
            res_dic[new_id] = sorted([("".join(core.split(" ")), len(votes) / len(value_lt),
                                       max(votes, key=lambda x: x[0])[0], [i[1] for i in votes], len(value_lt),
                                       sum([i[0] * (abs(len(value_lt) / 2 - i[1]) / len(value_lt)) for i in votes])) for
                                      core, votes in inter_dic.items()],
                                     key=lambda x: x[-1],
                                     reverse=True)[:numofget]
        return res_dic

    @staticmethod
    def _merge_res(sumdic):
        def merge(lt):
            lt = sorted(lt, key=lambda x: len(x[0]), reverse=True)
            res = []
            dead = []
            for i in range(len(lt)):
                if i in dead:
                    continue
                for j in range(i + 1, len(lt)):
                    if lt[j][0] in lt[i][0]:
                        core = lt[i][0]
                        votes = lt[i][1] + lt[j][1]
                        lt[i] = (core, votes)
                        dead.append(j)
                res.append(lt[i])

            return res

        def merge2dif(lt1, lt2, num):
            lt2 = sorted(lt2, key=lambda x: len(x[0]), reverse=True)
            res = []
            for i in lt2:
                is_dead = False
                for j in lt1:
                    if j[0] in i[0] or i[0] in j[0]:
                        is_dead = True
                        break
                if not is_dead:
                    res.append(i)
                if len(res) == num:
                    return res
            if len(res) > 0:
                return res
            else:
                return res.append(lt2[0])

        for new_id, votes in sumdic.items():
            # if new_id == "172f9dbc":
            #     print("debug")
            temp_a = []
            temp_b = []
            res_a = list()
            # if new_id == "172f9dbc":
            #     print(res_a)
            for i in votes:
                if i[2] > 0.1:
                    temp_a.append(i)
                else:
                    temp_b.append(i)
            if len(temp_a) <= 1:
                res_a.extend(temp_a)
            else:
                res_a = merge(temp_a)
            if len(res_a) >= 3 or len(temp_b) == 0:
                sumdic[new_id] = sorted(res_a, key=lambda x: x[1], reverse=True)[:3]
                continue
            else:
                b = merge2dif(temp_a, temp_b, 3 - len(res_a))
                try:
                    res_a.extend(b)
                except Exception as e:
                    print(e)
                sumdic[new_id] = res_a
            del res_a

    @staticmethod
    def _new_merge(dic):
        pass


    @staticmethod
    def from_bert_key_work_extract_get_res_to_txt_sord_and_clearn_them(model_output_file_base, out_file,
                                                                       raw_data_file_about_model_used,
                                                                       raw_data_is_train):
        data_util = from_txt_get_data.data_util()
        data_util.get_souhu_data(raw_data_file_about_model_used, raw_data_is_train)
        souhu_dic = data_util.souhu_data
        wf = open(out_file, "w", encoding="utf-8")
        file_names = os.listdir(model_output_file_base)
        for file_name in file_names:
            input_file = os.path.join(model_output_file_base, file_name)
            with open(input_file, "r") as rf:
                json_input = json.load(rf)
                temp = PostProcessTool._sentense_togeter(json_input)
                sum_dic = PostProcessTool._vote_and_sord(temp)
                # all_res = get_merge_dic(json_input)
                # sum_dic = get_sum_dic(all_res)
                PostProcessTool._amend_res_with_title(sum_dic, souhu_dic)
                # merge_res(sum_dic)

                # merge_title(sum_dic)
                fuse_dic = sum_dic
                for id, v in tqdm(fuse_dic.items()):
                    if raw_data_is_train:
                        core = [i[0] for i in [data[0] for data in souhu_dic[id].entity_mon]]
                    else:
                        core = ["PAD" for _ in range(3)]
                    # title = input_dic[id]["title"]
                    # emotion = [i[1] for i in input_dic[id]]

                    # try:
                    #     wf.write(
                    #         "1\t" +
                    #         id + "\t" + ",".join([v[0][0], v[1][0], v[2][0]]) + "\n")
                    #     wf.write(
                    #         "2\t" +
                    #         id + "\t" + ",".join(core) + "\n")
                    #     wf.write(
                    #         "3\t" + title + "\n"
                    #     )
                    # except Exception as e:
                    core_lt = []
                    core_vote = []
                    core_max_score = []
                    para_index = []
                    len_para = []
                    score = []
                    # emotion_lt = []
                    for index, i in enumerate(v):
                        core_lt.append(i[0])
                        score.append(i[-1])
                        core_vote.append(str(i[1]))
                        # try:
                        core_max_score.append(str(i[2]))
                        para_index.append(",".join([str(i) for i in i[3]]))
                        len_para.append(str(i[4]))
                        # except Exception as e:
                        #     print(e)
                        #     print("fjeiow")
                    # assert len(core_lt) == len(core_vote)
                    # assert len(core_vote) == len(core_max_score)
                    if len(core_lt) == 0:
                        print("fjewoa")
                    wf.write(
                        #     "1\t" +
                        id + "\t" + "*|||*".join(core_lt) + "\t" + "*|||*".join(core_vote) + "\t" + "*|||*".join(
                            core_max_score) + "\t" + "*|||*".join(para_index) + "\t" + "*|||*".join(
                            len_para) + "\t" + "*|||*".join(core) + "\n")
                    # wf.write(
                    # "1\t"+
                    # id + "\t" + "*|||*".join(core_lt) + "\t" + "*|||*".join(para_index) + "\t" + "*|||*".join(len_para) + "\t"+"*|||*".join(core) + "\t"+ "*|||*".join([str(i) for i in score])+ "\n")
                    # )
                    # wf.write(
                    #     "2\t" +
                    # id + "\t" +
                    # "*|||*".join(core) + "\n")
                    # wf.write(
                    #     "3\t" + title + "\n"
                    # )
                    # print("结果少于3个数为：{}".format(len(core_lt)))

    @staticmethod
    def get_core_res(extract_keywork_file, submission_output_file_base, write_file):
        sub_res = {}
        with open(extract_keywork_file, "r", encoding="utf-8") as rf:
            count = 0
            for line in rf:
                temp = line.strip().split("\t")
                sub_res[temp[0]] = temp[1].split("*|||*")
                if len(sub_res[temp[0]]) != 20:
                    mm = len(sub_res[temp[0]])
                    for _ in range(20 - len(sub_res[temp[0]])):
                        sub_res[temp[0]].append("[PAD]")
                    count += 1
                    print("count:{},len:{}".format(count, mm))

        wf = open(write_file, "w", encoding="utf-8")
        file_names = os.listdir(submission_output_file_base)
        for file_name in file_names:
            file_name = os.path.join(submission_output_file_base, file_name)
            with open(file_name, "r", encoding="utf-8") as rf:
                for line in rf:
                    temp = line.strip().split("\t")
                    tempa = [j[1] for j in
                             sorted([(i, index) for index, i in enumerate([float(mm) for mm in temp[1].split(",")]) if
                                     i >= 0.5],
                                    key=lambda x: x[0], reverse=True)]
                    if len(tempa) == 0:
                        temp[1] = [j[1] for j in
                                   sorted(
                                       [(i, index) for index, i in enumerate([float(mm) for mm in temp[1].split(",")])],
                                       key=lambda x: x[0], reverse=True)[:3]]
                    else:
                        temp[1] = tempa
                    # try:
                    res_lt = [sub_res[temp[0]][i] for i in temp[1] if sub_res[temp[0]][i] != "[PAD]"]
                    wf.write(temp[0] + "\t" + "*|||*".join(res_lt) + "\n")
                    # except Exception as e:
                    #     print(e)
            # wf.close()

    @staticmethod
    def f1score(res_file, label_file):
        res_dic = {}
        with open(res_file, "r", encoding="utf-8") as rf:
            for line in rf:
                temp = line.strip().split("\t")
                res_dic[temp[0]] = {"res": temp[1].split("*|||*")}
        with open(label_file, "r", encoding="utf-8") as rf:
            for line in rf:
                temp = line.strip().split("\t")
                res_dic[temp[0]]["label"] = temp[1].split("*|||*")
        score = []
        for id, data in res_dic.items():
            right_num = 0
            for res in data["res"]:
                if res in data["label"]:
                    right_num += 1
            one_score = right_num / len(data["label"])
            score.append(one_score)
        final_score = np.array(score).mean()
        print(score)
        print(final_score)

    @staticmethod
    def merge_core_emotion_finally(core_file, extract_kw_raw_output_base, emotion_file_base, writer_file):
        res_dic = {}
        with open(core_file, "r", encoding="utf-8")as rf:
            extract_kw_names = os.listdir(extract_kw_raw_output_base)
            for extract_kw_name in extract_kw_names:
                extract_kw_name = os.path.join(extract_kw_raw_output_base, extract_kw_name)
                nbest_dic = collections.defaultdict(list)
                nbest_json = json.load(open(extract_kw_name, "r", encoding="utf-8"))
                print(extract_kw_name)
                for k, v in tqdm(nbest_json.items()):
                    nbest_dic[k[:8]].append({k: [i["text"] for i in v]})
                for line in rf:
                    tempa = line.strip().split("\t")
                    dic_pos = []
                    for core in tempa[1].split("*|||*"):
                        if "," in core:
                            continue
                        core_post = []
                        for i in nbest_dic[tempa[0]]:
                            for k, v in i.items():
                                if " ".join(list(core)) in v:
                                    core_post.append(k)
                        dic_pos.append({core: core_post})
                    res_dic[tempa[0]] = dic_pos
        emotion_dic = {}
        num2emotion = {0: "POS", 1: "NORM", 2: "NEG"}
        emotion_names = os.listdir(emotion_file_base)
        for emotion_name in emotion_names:
            emotion_name = os.path.join(emotion_file_base, emotion_name)
            with open(emotion_name, "r", encoding="utf-8") as rf:
                print(emotion_name)
                for line in tqdm(rf):
                    temp = json.loads(line)
                    emotion_dic[temp["new_id"]] = temp["emotion"]
        final_res = collections.defaultdict(list)
        print("finaly result")
        for k, v in tqdm(res_dic.items()):
            for i in v:
                key, value = [mm for mm in i.items()][0]
                emotion_score = np.array([0, 0, 0])
                for j in value:
                    emotion_lt = np.array(emotion_dic[j])
                    emotion_score = emotion_score + emotion_lt
                final_res[k].append((key, num2emotion[np.argmax(emotion_score)]))
        with open(writer_file, "w", encoding="utf-8") as wf:
            print("begin writer")
            for k, v in tqdm(final_res.items()):
                wf.write(k + "\t" + ",".join([i[0] for i in v]) + "\t" + ",".join([i[1] for i in v]) + "\n")

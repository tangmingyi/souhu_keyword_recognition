from utility.data_tool.data_post_process import PostProcessTool
#拿到extract keywork的的row data 转变成 标准结果
PostProcessTool.from_bert_key_work_extract_get_res_to_txt_sord_and_clearn_them("data/res/extract_key_word_output","data/res/res_extract_keyword.txt","data/raw_data/test.txt",False)

#submission的raw data 到 标准结果。
PostProcessTool.get_finally_res("data/res/res_extract_keyword.txt","data/res/submission_output","data/res/all_core.txt")

#将emotion和core 结果合并。
PostProcessTool.merge_core_emotion_finally("data/res/all_core.txt","data/res/extract_key_word_output","data/res/emotion_output","data/res/result.txt")
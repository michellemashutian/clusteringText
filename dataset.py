#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19 20:24
# @Author  : Shutian
# @File    : dataset.py
# @Description    : loading txt
import jieba
import re


def ch_text_pro1(str1):
    # 针对 中文
    # 去除一些字符，替换多个空格为单个，字母取小写
    # For Chinese
    # Remove some characters, replace multiple spaces with a single one, and lowercase letters
    str2 = str1.replace("\\n", " ").replace("\u200b", "").replace("&amp;", "").replace("amp;", "").replace("quot;", "")
    remove_chars = '[!"#$%&()*+,-/:;<=>?@，。?★、…《》？?“”‘’！[\\]^_`{|}~—；▲！？｡＂＃＄％＆＇（）＊＋，－／：' \
                   '＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏]+'
    str3 = re.sub(remove_chars, ' ', str2)
    str4 = re.sub(' +', ' ', str3).strip()
    return str4


def text_seg(text, stopwords_file):
    # define the dict to store stopwords
    stop_words = {}
    with open(stopwords_file, "r", encoding='UTF-8') as f:
        for line in f:
            stop_words[line.strip()] = 1

    # text segmentation
    # remove strings like [XX]
    remove_chars = '\\[[\u4e00-\u9fa5]{2}\\]'
    text_pro = re.sub(remove_chars, '', text.strip().lower().replace("\t", " "))

    # text segmentation with jieba
    seg_list = jieba.cut(text_pro, cut_all=False)
    new_seg_list = []
    for x in seg_list:
        if x in stop_words:
            continue
        else:
            new_seg_list.append(x)
    seg_text = ch_text_pro1(" ".join(new_seg_list))
    return seg_text


class Dataset:

    def __init__(self, input_format):
        self.input_format = input_format

    def load_dataset(self, filename, stopword_filename):
        if self.input_format == "word":
            origin_data, process_data = self._load_word_dataset(filename)
            return origin_data, process_data
        elif self.input_format == "text":
            origin_data, process_data = self._load_text_dataset(filename, stopword_filename)
            return origin_data, process_data
        elif self.input_format == "word_text":
            origin_data, process_data = self._load_word_text_dataset(filename, stopword_filename)
            return origin_data, process_data
        else:
            raise Exception("format does not match!")

    def _load_word_dataset(self, filename):
        original_texts = []
        process_texts = []
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                if "\t" in line:
                    if len(line.split("\t")) == 2:
                        original_texts.append(line.strip())
                        doc_str = line.split("\t")[1]
                        process_texts.append(doc_str)
        return original_texts, process_texts

    def _load_text_dataset(self, filename, stopword_filename):
        original_texts = []
        process_texts = []
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                if "\t" in line:
                    if len(line.split("\t")) == 2:
                        original_texts.append(line.strip())
                        doc_str = line.split("\t")[1]
                        doc_seg = text_seg(doc_str, stopword_filename)
                        process_texts.append(doc_seg)
        return original_texts, process_texts

    def _load_word_text_dataset(self, filename, stopword_filename):
        original_texts = []
        process_texts = []
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                if "\t" in line:
                    if len(line.split("\t")) == 3:
                        original_texts.append(line.strip())
                        doc_str = line.split("\t")[1]
                        word_str = line.split("\t")[2]
                        doc_seg = text_seg(doc_str, stopword_filename)
                        process_texts.append(" ".join([word_str, doc_seg]))
        return original_texts, process_texts


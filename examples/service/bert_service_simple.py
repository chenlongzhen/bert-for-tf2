# bert 模型华为部署inference代码 v1
# by clz 20211216
# 参考 https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0301.html

import logging
import threading

import numpy as np
import tensorflow as tf
import os
from PIL import Image

from model_service.tfserving_model_service import TfServingBaseService
from .bert_tokenization import FullTokenizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class bert_service(TfServingBaseService):

    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.predict = None
        self.max_length = 256 #文本长度截断

        # label文件可以在这里加载,在后处理函数里使用
        # label.txt放在obs和模型包的目录

        # with open(os.path.join(self.model_path, 'label.txt')) as f:
        #     self.label = json.load(f)
        # 非阻塞方式加载saved_model模型，防止阻塞超时
        thread = threading.Thread(target=self.load_model)
        thread.start()
        
        # tokenizer
        self.tokenizer = FullTokenizer(vocab_file=os.path.join("vocab.txt")

    def load_model(self):
        """
        加载模型
        """
        self.model = tf.keras.models.load_model(self.model_path)
        self.predict = self.model.predict

    def _preprocess(self, data):
    
        # 截断处理
        max_seq_len_fix = self.max_length - 2  # cls sep词要减去
        processed_sentences = []
        for k, v in data.items():
            for file_name, text_content in v.items():
                if len(text_content) > max_seq_len_fix:
                    processed_sentences.append(text_content[:max_seq_len_fix])
                else:
                    processed_sentences.append(text_content)

        # tokenizer 将每个字id化
        pred_tokens = map(self.tokenizer.tokenize, processed_sentences)
        pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
        pred_token_ids = list(map(self.tokenizer.convert_tokens_to_ids, pred_tokens))
        pred_token_ids = map(lambda tids: tids + [0] * (max_seq_len_fix - len(tids)), pred_token_ids)
        pred_token_ids = np.array(list(pred_token_ids))
        
        return pred_token_ids

    def _inference(self, data):

        return self.predict(data)

    def _postprocess(self, data):

        return {
            "result": data[:,1] # predict的第二列是1的概率
        }
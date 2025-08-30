import re
import numpy as np

def getDataSet(dataPath=r"./SMSSpamCollection"):
    with open(dataPath, encoding='utf-8') as f:
        txt_data = f.readlines()
    # 所有邮件
    data = []
    # 标签
    classTag = []
    # 垃圾邮件
    spam_data_num = 0
    # 正常邮件
    ham_data_num = 0
    for line in txt_data:
        line_split = line.strip("\n").split('\t')
        if line_split[0] == "spam":
            data.append(line_split[1])
            spam_data_num += 1
            classTag.append(1)
        elif line_split[0] == "ham":
            data.append(line_split[1])
            ham_data_num += 1
            classTag.append(0)
    print("数据集大小为{}, 其中垃圾邮件数量为{}，正常邮件数量为{}".format(len(data), spam_data_num, ham_data_num))
    return data, classTag

class NaiveBayes:
    def __init__(self):
        self.spam_prob = None #垃圾短信概率
        self.ham_prob = None # 正常短信概率

        self.spam_word_prob = None # 词汇出现在垃圾短信的概率
        self.ham_word_prob = None # 词汇出现在正常短信的概率

        self.txt_word_set = set([]) # 词汇表
        self.word2index = {}

    # 输入为一则短信的内容
    def data_preprocess(self, txt_content):
        # 将输入转换为小写并将特殊字符替换为空格
        temp_info = re.sub(r'\W', ' ', txt_content.lower())
        # 根据空格将其分割为一个一个单词
        words = re.split(r'\s+', temp_info)
        # 返回长度大于等于3的所有单词
        return list(filter(lambda x: len(x) >= 3, words))

    def train(self, X, y):
        self.compute_spam_and_ham_prob(y)
        txt_word_list_set = [self.data_preprocess(txt_content) for txt_content in X]
        self.create_txt_word_set(txt_word_list_set)
        self.create_word_to_index(self.txt_word_set)

        spam_word_count_vector = np.zeros(len(self.txt_word_set), dtype=np.float64)
        ham_word_count_vector = np.zeros(len(self.txt_word_set), dtype=np.float64)
        for index, txt_word_list in enumerate(txt_word_list_set):
            if y[index] == 1:
                spam_word_count_vector += self.create_word_count_vector(txt_word_list)
            else:
                ham_word_count_vector += self.create_word_count_vector(txt_word_list)

        ## 拉普拉斯平滑过度, 计算词频, 取对数
        self.spam_word_prob = np.log((spam_word_count_vector + 1) / (np.sum(spam_word_count_vector) + len(self.txt_word_set)))
        self.ham_word_prob = np.log((ham_word_count_vector + 1) / (np.sum(ham_word_count_vector) + len(self.txt_word_set)))

    def predict(self, X):
        txt_word_list_set = [self.data_preprocess(txt_content) for txt_content in X]
        X_word_count_vector = np.zeros((len(X), len(self.txt_word_set)))
        for index, txt_word_list in enumerate(txt_word_list_set):
            X_word_count_vector[index] = self.create_word_count_vector(txt_word_list)

        spam_result = X_word_count_vector @ self.spam_word_prob + np.log(self.spam_prob)
        ham_result = X_word_count_vector @ self.ham_word_prob + np.log(self.ham_prob)
        result = (spam_result - ham_result > 0).astype(int)
        return result

    def create_word_count_vector(self, txt_word_list):
        word_count_vector = np.zeros(len(self.txt_word_set))
        for txt_word in txt_word_list:
            if self.word2index.get(txt_word, None) is not None:
                word_count_vector[self.word2index[txt_word]] += 1
            else:
                word_count_vector[self.word2index["<unknown>"]] += 1
        return word_count_vector

    def compute_spam_and_ham_prob(self, y):
        spam_count = np.sum(y)
        self.spam_prob = spam_count / len(y)
        self.ham_prob = 1 - self.spam_prob

    def create_word_to_index(self, txt_word_set):
        self.word2index = {word: i for i, word in enumerate(list(txt_word_set))}

    def create_txt_word_set(self, txt_word_list):
        for txt_content in txt_word_list:
            self.txt_word_set.update(txt_content)
        self.txt_word_set.add("<unknown>") # 其他不存在的词汇,默认映射到unknown

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

if __name__ == '__main__':
    txt_data_list, labels = getDataSet()
    labels = np.array(labels)

    X_train, X_test, y_train, y_test  = train_test_split(txt_data_list, labels, test_size=0.2, random_state=42)

    nb = NaiveBayes()
    nb.train(X_train, y_train)
    y_pred = nb.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

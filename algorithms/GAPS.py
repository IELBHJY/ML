import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


class GAPS:
    '''
    遗传算法超参数优化
    '''

    def __init__(self):
        self.train_data = None
        self.test_data = None
        # 遗传算法参数
        self._params_dict = dict()

        # 遗传算法数据结构
        self.pop_matrix = None
        self.children_matrix = None
        self.pop_scores = None
        self.sorted_item_info = None

    def _param_init(self, params_dict):
        if "param_len" not in params_dict.keys():
            return False
        else:
            if not isinstance(params_dict['param_len'], int):
                print("输入 param_len 参数 必须是 int 类型，且 > 0")
                return False
            self._params_dict['param_len'] = params_dict['param_len']
        if "param_width" not in params_dict.keys():
            self._params_dict['param_width'] = 8
        else:
            if not isinstance(params_dict['param_width'], int):
                print("输入 param_width 参数 必须是 int 类型，且 > 0")
                return False
            self._params_dict['param_width'] = params_dict['param_width']
        if "pop_size" not in params_dict.keys():
            self._params_dict['pop_size'] = 500
        else:
            self._params_dict['pop_size'] = params_dict['pop_size']
        if "mutate_rate" not in params_dict.keys():
            self._params_dict['mutate_rate'] = 0.4
        else:
            self._params_dict['mutate_rate'] = params_dict['mutate_rate']
        if "crossover_rate" not in params_dict.keys():
            self._params_dict['crossover_rate'] = 0.4
        else:
            self._params_dict['crossover_rate'] = params_dict['crossover_rate']
        if "inherit_rate" not in params_dict.keys():
            self._params_dict['inherit_rate'] = 0.2
        else:
            self._params_dict['inherit_rate'] = params_dict['inherit_rate']
        if "max_iter_num" not in params_dict.keys():
            self._params_dict['max_iter_num'] = 200
        else:
            self._params_dict['max_iter_num'] = params_dict['max_iter_num']
        if "problem_type" not in params_dict.keys():
            self._params_dict['problem_type'] = 'min'
        else:
            self._params_dict['problem_type'] = params_dict['problem_type']

    def _pop_init(self, init=None):
        self.pop_matrix = np.random.randint(0, 2, [self._params_dict['pop_size'],
                                                   self._params_dict['param_width'] * self._params_dict['param_len']],
                                            dtype=int)
        if init is not None:
            for i in range(0, init.shape[0]):
                self.pop_matrix[i] = init[i]
        self.children_matrix = np.zeros([self._params_dict['pop_size'],
                                         self._params_dict['param_width'] * self._params_dict['param_len']],
                                        dtype=int)
        self.pop_scores = dict()
        for i in range(self._params_dict['pop_size']):
            self.pop_scores[i] = 0.0

    def _update(self, train, test, obj, feval, item_index):
        # 解码，获取参数实际值
        item = self.pop_matrix[item_index]
        param_values = np.zeros([self._params_dict['param_len']], dtype=float)
        for i in range(self._params_dict['param_len']):
            param_value = 0
            for j in range(self._params_dict['param_width']):
                param_value += item[i * self._params_dict['param_width'] + j] << j
            param_value = param_value / 100.0
            param_values[i] = param_value
        # obj feval 形式待定
        pre_test = obj(train, test, param_values)
        acc5, acc10, new_acc = feval(test, pre_test)
        return new_acc

    def _cal_pop_score(self, train, test, obj, feval):
        for i in range(self._params_dict['pop_size']):
            self.pop_scores[i] = self._update(train=train, test=test, obj=obj, feval=feval, item_index=i)

        reverse_flag = (True if self._params_dict['problem_type'] == 'max' else False)
        self.sorted_item_info = sorted(self.pop_scores.items(), key=lambda x: x[1], reverse=reverse_flag)

    def _choose_item(self):
        score_sum = sum(self.pop_scores.values())
        value = np.random.randint(0, int(score_sum))
        index = 0
        cumsum = self.pop_scores[index]
        while cumsum < value:
            index += 1
            cumsum += self.pop_scores[index]
        return index

    def _crossover(self, index1, index2, new_index1, new_index2):
        cv_point = np.random.randint(0, int(self._params_dict['param_width'] * self._params_dict['param_len']))
        item1 = np.array(np.hstack((self.pop_matrix[index1, :cv_point],
                                    self.pop_matrix[index2, cv_point:])), dtype=int)
        item2 = np.array(np.hstack((self.pop_matrix[index2, :cv_point],
                                    self.pop_matrix[index1, cv_point:])), dtype=int)
        self.children_matrix[new_index1] = item1
        self.children_matrix[new_index2] = item2

    def _mutation(self, index, new_index):
        item = np.array(self.pop_matrix[index], dtype=int)
        for i in range(self._params_dict['param_len']):
            j = np.random.randint(0, self._params_dict['param_width'])
            item[i + j] = 1 - item[i + j]
        self.children_matrix[new_index] = item

    def _evolutionary_operator(self):
        # 先保留精英
        start_index = 0
        end_index = int(self._params_dict['pop_size'] * self._params_dict['inherit_rate'])
        for i in range(start_index, end_index, 1):
            self.children_matrix[i] = np.array(self.pop_matrix[self.sorted_item_info[i][0]], dtype=int)

        # 交叉
        start_index = end_index
        end_index = start_index + int(self._params_dict['pop_size'] * self._params_dict['crossover_rate'])
        for i in range(start_index, end_index, 2):
            index1 = self._choose_item()
            index2 = self._choose_item()
            while index1 == index2:
                index2 = self._choose_item()
            self._crossover(index1, index2, i, i + 1)

        # 变异
        start_index = end_index
        for i in range(start_index, self._params_dict['pop_size']):
            index = self._choose_item()
            self._mutation(index, i)

    def _exchange(self):
        self.pop_matrix = np.array(self.children_matrix, dtype=int)

    def _genetic_algorithm_process(self, train, test, obj, feval):
        iter = 0
        while iter < self._params_dict['max_iter_num']:
            self._evolutionary_operator()
            self._exchange()
            self._cal_pop_score(train, test, obj, feval)
            iter += 1
            if iter % 100 == 0:
                print("最好的目标函数值:{}".format(self.sorted_item_info[0][1]))

    def _output(self):
        best_item = self.pop_matrix[self.sorted_item_info[0][0]]
        obj_value = self.sorted_item_info[0][1]
        param_values = np.zeros([self._params_dict['param_len']], dtype=float)
        for i in range(self._params_dict['param_len']):
            param_value = 0
            for j in range(self._params_dict['param_width']):
                param_value += best_item[i * self._params_dict['param_width'] + j] << j
            param_value = param_value / 100.0
            param_values[i] = param_value
        return param_values, obj_value

    def _fit_param_internal(self, train, test, params_dict, obj, feval, init):
        # 1、初始化
        self._param_init(params_dict=params_dict)
        self._pop_init()
        self._cal_pop_score(train, test, obj, feval)

        # 2、GA 主流程
        self._genetic_algorithm_process(train, test, obj, feval)

        # 3、求解结束
        return self._output()

    def fit_param(self, train_data, test_data, params_dict, obj=None, feval=None, init_solution=None):
        '''
        train_data:训练数据集
        test_data:验证数据集
        obj:user define train function
        feval:user define metrics function
        '''
        if obj is None or feval is None:
            assert "必须输入训练函数和评价函数～"
            return None
        else:
            return self._fit_param_internal(train_data, test_data, params_dict, obj=obj, feval=feval, init=init_solution)


if __name__ == '__main__':
    data = load_boston()
    X, y = data.data, data.target
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=2020)
    train_y = train_y.reshape(train_y.shape[0], 1)
    train_data = np.hstack((train_x, train_y))
    test_y = test_y.reshape(test_y.shape[0], 1)
    test_data = np.hstack((test_x, test_y))


    def user_train_function(train_data, test_data, param_values):
        test = test_data[:, 0:13]
        p = np.array(param_values)
        return (test @ p.T).reshape(1, -1)[0]


    def user_feval_function(test_data, test_pre):
        test_y = test_data[:, 13]
        return 1, 1, sum(abs(test_y - test_pre))


    gaps = GAPS()
    param = dict()
    param['pop_size'] = 500
    param['param_len'] = 13
    param['param_width'] = 4
    param['problem_type'] = 'min'
    param['max_iter_num'] = 500
    sol, obj_value = gaps.fit_param(train_data, test_data, params_dict=param,
                                    obj=user_train_function,
                                    feval=user_feval_function)
    print(sol)
    print(obj_value)
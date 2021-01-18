import numpy as np


class GAPS:
    """Evolutionary search of best hyperparameters, based on Genetic
    Algorithms

    Parameters
    ----------
    params : dict
    con_param: 连续参数，需要给定取值范围
    dis_param: 离散参数，需要给定取值

    Examples
    --------
    import warnings
    from sklearn.datasets import load_boston
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    data = load_boston()
    X, y = data.data, data.target
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=2020)
    train_y = train_y.reshape(train_y.shape[0], 1)
    train_data = np.hstack((train_x, train_y))
    test_y = test_y.reshape(test_y.shape[0], 1)
    test_data = np.hstack((test_x, test_y))

    def user_train_function(train_x, train_y, test_x, param_values):
        n_estimators = param_values['n_estimators']
        lr = param_values['learning_rate'] / 10.0
        subsample = param_values['subsample'] / 10.0
        max_depth = param_values['max_depth']
        min_samples_split = param_values['min_samples_split'] / 10.0
        min_samples_leaf = param_values['min_samples_leaf'] / 10.0
        if min_samples_leaf == 1:
            min_samples_leaf = int(min_samples_leaf)
        gbdt = GradientBoostingRegressor(
            learning_rate=lr,
            n_estimators=n_estimators,
            subsample=subsample,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=2021
        )
        gbdt.fit(train_x, train_y)
        y_pre = gbdt.predict(test_x)
        return y_pre


    def user_feval_function(test_y, test_pre):
        return mean_absolute_error(test_y, test_pre)

    con_param = dict()
    con_param['n_estimators'] = [30, 100]

    dis_param = dict()
    dis_param['subsample'] = [2, 5, 6, 7, 10]
    dis_param['max_depth'] = [2, 3, 4, 5]
    dis_param['min_samples_split'] = [2, 4, 5, 8, 10]
    dis_param['min_samples_leaf'] = [2, 3, 4, 10]
    dis_param['learning_rate'] = [1, 2, 3, 4, 5, 6, 7, 8]
    gaps = GAPS(continue_param_dict=con_param, dispersion_param_dict=dis_param)
    sol, obj_value = gaps.search(train_x,
                                 train_y,
                                 test_x,
                                 test_y,
                                 training=user_train_function,
                                 scoring=user_feval_function)


    Attributes
    ----------
    sol 最优的参数取值
    obj_value 对应的score
    """

    def __init__(self,
                 continue_param_dict=None,
                 dispersion_param_dict=None,
                 fit_param=None
                 ):
        self.train_data = None
        self.test_data = None
        self.error_info = None
        self.init_success = True
        self.continue_param_len = 0
        self.dispersion_param_len = 0
        self.continue_param_dict = continue_param_dict
        self.dispersion_param_dict = dispersion_param_dict
        self.continue_param_names = [] if continue_param_dict is None else list(continue_param_dict.keys())
        self.dispersion_param_names = [] if dispersion_param_dict is None else list(dispersion_param_dict.keys())
        self._fit_param = dict() if fit_param is None else fit_param
        self._params_dict = dict()
        self._param_code_pos = dict()
        self._param_code_name = dict()

        # 遗传算法需要优化的参数
        self._optimize_param(continue_param_dict, dispersion_param_dict)
        if not self.init_success:
            print("Init failed.Error info:{}".format(self.error_info))

        # 遗传算法参数
        self._param_init(self._fit_param)
        self._check_param()
        if not self.init_success:
            print("Init failed.Error info:{}".format(self.error_info))

        # 遗传算法数据结构
        self.pop_matrix = None
        self.children_matrix = None
        self.pop_scores = None
        self.sorted_item_info = None

    def _check_param(self):
        pass

    def _param_init(self, params_dict):
        if "pop_size" not in params_dict.keys():
            self._params_dict['pop_size'] = 100
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
            self._params_dict['max_iter_num'] = 30
        else:
            self._params_dict['max_iter_num'] = params_dict['max_iter_num']
        if "problem_type" not in params_dict.keys():
            self._params_dict['problem_type'] = 'min'
        else:
            self._params_dict['problem_type'] = params_dict['problem_type']

    def _optimize_param(self, continue_param_dict, dispersion_param_dict):
        if continue_param_dict is None and dispersion_param_dict is None:
            self.init_success = False
            self.error_info = "Input optimize param is None."
            return
        self._params_dict['param_len'] = 0
        i = 0

        if not (continue_param_dict is None):
            if not isinstance(continue_param_dict, dict):
                self.init_success = False
                self.error_info = "Input continue_param_dict is not dict."
                return
            if len(continue_param_dict) == 0:
                return
            self.continue_param_len = len(continue_param_dict)
            self._params_dict['param_len'] += len(continue_param_dict)
            max_value = -np.inf
            min_value = np.inf
            for key in continue_param_dict.keys():
                item = continue_param_dict[key]
                if item[0] < min_value:
                    min_value = item[0]
                if item[1] > max_value:
                    max_value = item[1]
            param_width = 1
            cumvalue = 1
            while cumvalue < max_value:
                cumvalue += 1 << param_width
                param_width += 1
            self._params_dict['param_width'] = param_width
            # 确定每一个参数，基因的位置
            for key in continue_param_dict.keys():
                # 计算左界的位置
                param_pos = 0
                cumvalue = 1
                while cumvalue < continue_param_dict[key][0]:
                    param_pos += 1
                    cumvalue += 1 << param_pos
                item = [param_pos]

                # 计算右界的位置
                param_pos = 0
                cumvalue = 1
                while cumvalue < continue_param_dict[key][1]:
                    param_pos += 1
                    cumvalue += 1 << param_pos
                item.append(param_pos)
                self._param_code_pos[key] = item
                self._param_code_name[i] = key
                i += 1

        if not (dispersion_param_dict is None):
            if not isinstance(dispersion_param_dict, dict):
                self.init_success = False
                self.error_info = "Input dispersion_param_dict is dict."
            if len(dispersion_param_dict) == 0:
                return
            self.dispersion_param_len = len(dispersion_param_dict)
            self._params_dict['param_len'] += len(dispersion_param_dict)
            # TODO 获取离散变量的长度，是否超出当前参数宽度所能表示的范围

            for key in dispersion_param_dict.keys():
                param_pos = 0
                item = [param_pos]
                param_pos = len(dispersion_param_dict[key]) - 1
                item.append(param_pos)
                self._param_code_pos[key] = item
                self._param_code_name[i] = key
                i += 1

    def _pop_init(self, init=None):
        self.pop_matrix = np.random.randint(0, 2, [self._params_dict['pop_size'],
                                                   self._params_dict['param_width'] * self._params_dict['param_len']],
                                            dtype=int)
        if init is not None:
            pass

        self.children_matrix = np.zeros([self._params_dict['pop_size'],
                                         self._params_dict['param_width'] * self._params_dict['param_len']],
                                        dtype=int)
        self.pop_scores = dict()
        for i in range(self._params_dict['pop_size']):
            self.pop_scores[i] = 0.0

    def _uncode(self, item):
        # 解码 返回实际参数值
        param_values = dict()
        for i in range(self._params_dict['param_len']):
            param_value = 0
            name = self._param_code_name[i]
            for j in range(self._params_dict['param_width']):
                if self._param_code_pos[name][0] <= j <= self._param_code_pos[name][1]:
                    param_value += item[i * self._params_dict['param_width'] + j] << j

            if i < self.continue_param_len:
                param_name = self.continue_param_names[i]
                if param_value > self.continue_param_dict[param_name][1]:
                    param_value = self.continue_param_dict[param_name][1]
                if param_value < self.continue_param_dict[param_name][0]:
                    param_value = self.continue_param_dict[param_name][0]
                param_values[param_name] = param_value
            else:
                param_name = self.dispersion_param_names[i - self.continue_param_len]
                if param_value >= len(self.dispersion_param_dict[param_name]) - 1:
                    param_value = len(self.dispersion_param_dict[param_name]) - 1
                param_value = self.dispersion_param_dict[param_name][param_value]
                param_values[param_name] = param_value
        return param_values

    def _update(self, train_x, train_y, test_x, test_y, obj, feval, item_index):
        # 获取参数实际值
        item = self.pop_matrix[item_index]
        param_values = self._uncode(item)
        pre_test = obj(train_x, train_y, test_x, param_values)
        score = feval(test_y, pre_test)
        return score

    def _cal_pop_score(self, train_x, train_y, test_x, test_y, obj, feval):
        for i in range(self._params_dict['pop_size']):
            self.pop_scores[i] = self._update(train_x=train_x,
                                              train_y=train_y,
                                              test_x=test_x,
                                              test_y=test_y,
                                              obj=obj, feval=feval, item_index=i)

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

    def _genetic_algorithm_process(self, train_x, train_y, test_x, test_y, obj, feval):
        iter = 0
        while iter < self._params_dict['max_iter_num']:
            self._evolutionary_operator()
            self._exchange()
            self._cal_pop_score(train_x, train_y, test_x, test_y, obj, feval)
            iter += 1
            if iter % 10 == 0:
                print("最好的目标函数值:{}".format(self.sorted_item_info[0][1]))

    def _output(self):
        best_item = self.pop_matrix[self.sorted_item_info[0][0]]
        obj_value = self.sorted_item_info[0][1]
        param_values = self._uncode(best_item)
        return param_values, obj_value

    def _search_internal(self, train_x, train_y, test_x, test_y, obj, feval, init=None):
        # 1、初始化
        self._pop_init(init=init)
        self._cal_pop_score(train_x, train_y, test_x, test_y, obj, feval)

        # 2、GA 主流程
        self._genetic_algorithm_process(train_x, train_y, test_x, test_y, obj, feval)

        # 3、求解结束
        return self._output()

    def search(self,
               train_x=None,
               train_y=None,
               test_x=None,
               test_y=None,
               training=None,
               scoring=None,
               init_solution=None
               ):
        if training is None or scoring is None:
            assert "必须输入训练函数和评价函数～"
            return None
        else:
            return self._search_internal(train_x,
                                         train_y,
                                         test_x,
                                         test_y,
                                         obj=training,
                                         feval=scoring,
                                         init=init_solution)

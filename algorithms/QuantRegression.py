import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import os


class QR:
    def __init__(self):
        self.res =None
        self.temp =None

    def load_data(self, data_path):
        self.temp = pd.read_csv(data_path)

    def fit(self, y_name, features_name):
        if np.linalg.matrix_rank(self.temp[['mileage', 'weight_mileage']]) == 1:
            return None
        if self.temp is not None:
            try:
                model = smf.quantreg(y_name + '~' + features_name, self.temp)
                res = model.fit(q=.5, max_iter=10000)
                self.res = res
            except:
                self.res = None
                return None
        if self.res.params['mileage'] <= 0 or \
                self.res.params['weight_mileage'] <= 0 or \
                self.res.params['Intercept'] <= 0:
            dirs = 'output/QR/' + str(self.start_cityid) + '/no/'
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            self.show_result(dirs)
            return None
        dirs = 'output/QR/' + str(self.start_cityid) + '/yes/'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        self.show_result(dirs)
        return self.res


if __name__ == "__main__":
    quantReg = QR()
    quantReg.load_data('data.csv')
    res = quantReg.fit('price', 'mileage + weight_mileage')

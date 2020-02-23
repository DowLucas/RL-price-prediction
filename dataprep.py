import json
import numpy as np


class DataPreperator():
    def remove_empty_items(self, di):
        di = {key: d for key, d in di.items() if len(d) != 0}
        return di

    def minimum_length_keep(self, min_length, di):
        delete_list = []
        for item, d in di.items():
            for key, prices in d.items():
                if len(prices) < min_length:
                    delete_list.append([item, key])

        for de in delete_list:
            del di[de[0]][de[1]]
        return di

    def replace_zeroes(self, arr):
        for n, val in enumerate(arr):
            if val == 0:
                arr[n] = [0 for _ in range(6)]
        return arr

    def remove_arrays_with_many_zeroes(self, di, porportion_allowed=0.5):
        non_zero_porportion_allowed = 1-porportion_allowed
        delete_list = []
        for item, d in di.items():
            for key, prices in d.items():
                list_len = len(prices)
                if np.count_nonzero(prices) < np.floor(non_zero_porportion_allowed*list_len):
                    delete_list.append([item, key])
                else:
                    di[item][key] = self.replace_zeroes(prices)

        for de in delete_list:
            del di[de[0]][de[1]]
        return di

    # Sequence length, 10 values and 11th in the prediction
    def create_all_data(self, di, train_list_len, sequence_length = 10):
        train_x = []
        train_y = []
        train_index = 0
        for item, d in di.items():
            startIndex = 0
            for key, prices in d.items():
                endIndex = startIndex + sequence_length

                if len(prices) == endIndex:
                    break

                series = np.array(prices[startIndex:endIndex])
                target = np.array(prices[endIndex+1])

                train_x.append(series)
                train_y.append(target)

                #print(train_x, train_y)


                train_index += 1


        return np.array(train_x, dtype=np.float32), np.array(train_y, dtype=np.float32)

data = json.load(open("data.json", "r"))
dp = DataPreperator()

data = dp.minimum_length_keep(20, data)
data = dp.remove_arrays_with_many_zeroes(data, 0.5)
data = dp.remove_empty_items(data)

X, y = dp.create_all_data(data, len(data))


np.save("X.npy", X)
np.save("y.npy", y)




#print(data)
print(len(data))
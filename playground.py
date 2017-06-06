import numpy as np
import time
import random

numbers = random.sample(list(range(1,10000)), 9999)

f = np.array(numbers)

f -= np.max(f)

elapsed_time_1 = elapsed_time_2 = 0.0

for i in range(1, 200):

    start_time = time.monotonic()
    p = np.exp(f) / np.sum(np.exp(f))
    elapsed_time_1 += time.monotonic() - start_time

    # print(p)
    # print('1).', elapsed_time_1)

    start_time = time.monotonic()
    f_exp = np.exp(f)
    f_sum = np.sum(f_exp)
    p = f_exp / f_sum
    elapsed_time_2 += time.monotonic() - start_time

    # print(p)
    # print('2).', elapsed_time_2)


print('1).', elapsed_time_1 / 200)
print('2).', elapsed_time_2 / 200)
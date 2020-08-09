from datetime import datetime
import random
import time
odds = [1,2,3,4,5,6,7,8,9,10,11,
        13,19,20,14,15,28,29,
        16,17,18,32,33,44,55,66,77,89]

for i in range(5):
    right_this_minute = datetime.today().minute

    if right_this_minute in odds:
        print('a little odds' + str(right_this_minute))
    else:
        print('not an odds')
    wait_time = random.randint(1,60)
    print(wait_time)
    time.sleep(wait_time)
import os
import datetime
import threading
import random

PATH = '/data/profile/network_logs/memory_usage.log'

last_used = -100.0
last_free = -100.0

def run(sec):
    global last_used
    global last_free
    t = threading.Timer(sec, run, [sec])
    tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    ratio_used = used_m * 100.0 / tot_m
    ratio_free = free_m * 100.0 / tot_m
    if abs(ratio_used - last_used) > 0.1 or abs(ratio_free - last_free) > 0.1:
        last_used = ratio_used
        last_free = ratio_free
        with open(PATH, 'a+') as f:
            f.write(f'[{str(datetime.datetime.now())}] Used: {ratio_used:.2f}% | Free: {ratio_free:.2f}%\n')
    t.start()

if __name__ == '__main__':

    run(3.0)

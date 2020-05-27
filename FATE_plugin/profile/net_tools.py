import os
import json
import fcntl
import datetime
from pympler import asizeof

OUTPUT_PATH = '/data/profile/network_logs/latest'
LOGS_PATH = '/data/profile/network_logs/latest/logs.txt'

def sz(size):
    units = ['b', 'kb', 'mb', 'gb', 'tb']
    unit_index = 0
    while size >= 1024.0 and unit_index < 4:
        unit_index += 1
        size /= 1024.0
    return f'{size:.3f} {units[unit_index]}'

class netstat:

    def __init__(self, job_id, job_name, action, tag, src, src_role, dst, dst_role):
        """
        Add a log record to preset file
        :param job_id: id of the overall job
        :param job_name: name of the transmission job
        :param action: can be "recv" or "send"
        :param tag: identifier of the transmission job
        :param src: initiator of the transmission
        :param src_role: role of the initiator
        :param dst: receiver of the transmission
        :param dst_role: role of the receiver
        """
        self.job_id = job_id
        self.job_name = job_name
        self.action = action
        self.tag = tag
        self.src = f'[{src_role}] {src}'
        self.dst = f'[{dst_role}] {dst}'
        self.size = 0

    def tick(self):
        self.start = datetime.datetime.now()

    def tock(self):
        self.end = datetime.datetime.now()

    def add_size(self, obj):
        self.size += asizeof.asizeof(obj)

    def add_log(self):

        filename = os.path.join(OUTPUT_PATH, f'{self.job_id}.log')
        lockname = os.path.join(OUTPUT_PATH, f'{self.job_id}.lock')

        if not os.path.exists(lockname):
            with open(lockname, 'w') as f:
                f.write('')

        with open(lockname, 'r') as lock:
            
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

            if not os.path.exists(filename):
                with open(filename, 'w') as f:
                    init_data = {
                        'job' : self.job_id,
                        'logs' : [],
                    }
                    json.dump(init_data, f)

            data = None

            with open(filename, 'r') as f:
                data = json.load(f)

            with open(filename, 'w') as f:
                det = {
                    'job_name' : str(self.job_name),
                    'action' : str(self.action),
                    'tag' : str(self.tag),
                    'src' : str(self.src),
                    'dst' : str(self.dst),
                    'start' : str(self.start),
                    'elapse' : str(self.end - self.start),
                    'size' : sz(self.size),
                    'bsize' : self.size
                }
                data['logs'].append(det)
                json.dump(data, f)
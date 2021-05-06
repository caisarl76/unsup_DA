from datetime import datetime
from pytz import timezone


class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def write(self, log, end='\n', is_print=True, add_time=True):
        if add_time:
            log = Log.add_time(log)
        if is_print:
            print(log, end=end)
        with open(self.log_path, 'a') as f:
            f.write(log + end)

    @staticmethod
    def add_time(log):
        fmt = "%Y-%m-%d %H:%M:%S %Z%z"
        kst_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
        time_log = '%s: %s' % (kst_time, log)

        return time_log
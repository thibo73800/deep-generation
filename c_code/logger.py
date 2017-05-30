import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
file_handler = RotatingFileHandler('c_code.log', 'a', 1000000, 1)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
steam_handler = logging.StreamHandler()
steam_handler.setLevel(logging.DEBUG)
logger.addHandler(steam_handler)

class Logger(object):
    def __init__(self, label):
        super(Logger, self).__init__()
        self.label = label
        self.logger = logger

    def debug(self, string):
        self.logger.debug("%s::%s" % (self.label, string))

    def info(self, string):
        self.logger.info("%s::%s" % (self.label, string))

    def warning(self, string):
        self.logger.warning("%s::%s" % (self.label, string))

    def error(self, string):
        self.logger.error("%s::%s" % (self.label, string))

    def critical(self, string):
        self.logger.critical("%s::%s" % (self.label, string))

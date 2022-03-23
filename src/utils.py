import logging
import os


class CustomFormatter(logging.Formatter):
    blue = "\x1b[34m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    green = "\x1b[32m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(log_path, logfilenames, log_name=None, displaying=True, saving=True, level="INFO"):
    log_dir = os.path.dirname(log_path)
    makedirs(log_dir)
    log_file = os.path.basename(log_path)
    if log_file == "":
        if log_name is None:
            lst = [os.path.basename(file).split(".", 1)[0] for file in logfilenames]
            log_file = "_".join(lst)
        else:
            log_file = log_name
    logpath = os.path.join(log_dir, log_file)
    logger = logging.getLogger()

    # To get rid of multiple adding of handlers from a consecutive calls to loggers
    while logger.handlers:
        logger.handlers.pop()

    level = logging.getLevelName(level)
    logger.setLevel(level)

    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(CustomFormatter())
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)

    return logger, logpath, log_dir

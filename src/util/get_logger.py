import logging


def get_logger(file_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(
        '%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]:\n%(message)s'
    )
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]:\n%(message)s'
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False
    
    return logger
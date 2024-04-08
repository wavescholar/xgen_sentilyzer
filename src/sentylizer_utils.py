import logging


def logging_config(base_data_directory, now):

    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    log_file_name = base_data_directory + "/wavelang_{:%Y-%m-%d-%H-%M}.log".format(now)

    logging.basicConfig(
        filename=log_file_name,
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(lineno)04d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logging.debug("wavelang log file ")
    logging.info("date and time =" + dt_string)
    logging.info("---------" + dt_string + "----------")

    return logger

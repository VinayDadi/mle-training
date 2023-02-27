import logging
import logging.config
import os

LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",  # noqa:E501
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(message)s"},
    },
    # "handlers": {"sh": {"class": "logging.StreamHandler", "formatter": "default"}},
    "root": {"level": "DEBUG"},
}


def configure_logger(
    logger=None, cfg=None, log_file=None, console=True, log_level="DEBUG"
):
    """Function to setup configurations of logger through function.

    The individual arguments of `log_file`, `console`, `log_level` will overwrite the ones in cfg.

    Parameters
    ----------
            logger:
                    Predefined logger object if present. If None a ew logger object will be created from root.
            cfg: dict()
                    Configuration of the logging to be implemented by default
            log_file: str
                    Path to the log file for logs to be stored
            console: bool
                    To include a console handler(logs printing in console)
            log_level: str
                    One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
                    default - `"DEBUG"`

    Return
    -------
    logging.Logger
    """  # noqa:E501
    if not cfg:
        logging.config.dictConfig(LOGGING_DEFAULT_CONFIG)
    else:
        logging.config.dictConfig(cfg)

    info_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s:%(lineno)d - %(message)s",  # noqa:E501
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    simple_format = logging.Formatter("%(message)s")

    logger = logger or logging.getLogger(__name__)

    if log_file or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_file:
            if not os.path.exists(os.path.dirname(log_file)):
                os.makedirs(os.path.dirname(log_file))
            fh = logging.FileHandler(log_file)
            fh.setFormatter(info_format)
            fh.setLevel(getattr(logging, log_level))
            logger.addHandler(fh)

        if console:
            sh = logging.StreamHandler()
            sh.setFormatter(simple_format)
            sh.setLevel(getattr(logging, log_level))
            logger.addHandler(sh)

    return logger


if __name__ == "__main__":
    logger = configure_logger()
    logger.debug("Running Logger")

import logging
import wandb
########################################
# Import Logging
########################################
def LoggerSetting():
    """
    Set the Logger
    Returns:
        log
    """
    logging.getLogger('PIL').setLevel(logging.WARNING)
    # Logging
    logger = logging.getLogger()
    # 로깅 레벨을 낮춰서 디버그 모드일 때 로깅 내역을 출력하도록 한다.
    logger.setLevel(logging.ERROR)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s(%(name)s): %(message)s')
    consH = logging.StreamHandler()
    consH.setFormatter(formatter)
    consH.setLevel(logging.DEBUG)
    logger.addHandler(consH)
    # filehandler = logging.FileHandler(f'{opt.outf}_logfile.log')
    # logger.addHandler(filehandler)
    log = logger
    return log
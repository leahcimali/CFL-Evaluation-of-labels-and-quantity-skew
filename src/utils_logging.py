import logging
import os

def setup_logging(output_name: str = "info"):
    """
    Set up logging with a unique log file for each experiment.

    Arguments:
        output_name (str): Name of the log file (typically experiment-specific).
    """

    # Ensure log directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log file for each experiment
    log_file = os.path.join(log_dir, f"{output_name}.log")

    # Define file handler
    file_handler = logging.FileHandler(log_file, mode="a")
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    file_handler.setFormatter(formatter)

    # Define console handler for real-time logging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Get logger and configure handlers
    logger = logging.getLogger("main_log")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers in case of multiple calls
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)


def cprint(msg: str, lvl: str = "info") -> None:
    """
    Print message to the console and log file at the desired logging level.

    Arguments:
        msg (str): Message to print.
        lvl (str): Logging level ("debug", "info", "warning", "error", "critical").
    """
    logger = logging.getLogger("main_log")

    if lvl == "debug":
        logger.debug(msg)
    elif lvl == "info":
        logger.info(msg)
    elif lvl == "warning":
        logger.warning(msg)
    elif lvl == "error":
        logger.error(msg)
    elif lvl == "critical":
        logger.critical(msg)


if __name__ == "__main__":
    setup_logging()  # Default log file: info.log
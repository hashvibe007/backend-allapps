import logging


def get_logger(name: str = __name__):
    """Get a configured logger instance."""
    logging.basicConfig(
        filename="processing_engine.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(name)

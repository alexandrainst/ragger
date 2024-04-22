"""Script to run a local development GUI.

Usage:
    python src/scripts/run_gui.py <key>=<value> <key>=<value> ...
"""

import logging
import os

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from ragger.gui import build_gui

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


@hydra.main(config_path="../../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the GUI.

    Args:
        cfg:
            The Hydra configuration.
    """
    gui = build_gui(cfg=cfg)
    logger.info(f"GUI started on http://{cfg.host}:{cfg.port}.")

    # Set up authentication
    if cfg.share and cfg.password_protected:
        username = os.getenv("GRADIO_USERNAME", "")
        password = os.getenv("GRADIO_PASSWORD", "")
        if not username:
            logger.warning("No username provided. Using empty string.")
        if not password:
            logger.warning("No password provided. Using empty string.")
        auth = (username, password)
    else:
        auth = None

    gui.queue().launch(
        share=cfg.share, auth=auth, server_name=cfg.host, server_port=cfg.port
    )


if __name__ == "__main__":
    main()

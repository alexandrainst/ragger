"""Script to run a local demo.

Usage:
    python src/scripts/run_demo.py <key>=<value> <key>=<value> ...
"""

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from ragger.demo import Demo

load_dotenv()


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Run the GUI.

    Args:
        config:
            The Hydra configuration.
    """
    demo = Demo(config=config)
    demo.launch()


if __name__ == "__main__":
    main()

"""Console output: rich logging and one shared console (progress displays and tables print through it)."""
import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup(verbosity=0):
    """verbosity: -1 quiet (warnings only), 0 info, 1 debug."""
    level = {-1: logging.WARNING, 0: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(level=level, format="%(message)s", datefmt="[%X]", force=True,
                        handlers=[RichHandler(console=console, show_path=False, markup=True,
                                              rich_tracebacks=True, log_time_format="[%X]")])
    for noisy in ("websockets", "asyncio", "matplotlib", "PIL", "h5py"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.captureWarnings(True)


def viser_server(port):
    """A viser server without its start-up banner (we log the URL ourselves)."""
    import rich
    import viser
    with rich.get_console().capture():
        return viser.ViserServer(port=port, verbose=False)

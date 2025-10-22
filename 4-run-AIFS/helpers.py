import logging
import datetime
import numpy as np

LOGGER = logging.getLogger(__name__)

def load_saved_state(file) -> dict:
    # Get the date from the filename: f"inputstate-{date.strftime('%Y%m%d_%H')}.npz"
    date_str = file.stem.split("-")[1]
    date = datetime.datetime.strptime(date_str, "%Y%m%d_%H")

    with np.load(file, allow_pickle=False) as data:
        fields = {k: data[k] for k in data.files}

    state = {"date": date, "fields": fields}
    return state


def save_state(state, outfile):
    np.savez(outfile, **state["fields"])

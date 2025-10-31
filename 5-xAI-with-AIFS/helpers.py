import datetime
import logging
from collections import defaultdict
from pathlib import Path

import earthkit.data as ekd
import earthkit.regrid as ekr
import numpy as np
import pandas as pd
import torch

LOGGER = logging.getLogger(__name__)

R = 6371.0  # Earth radius in km
GRID_RESOLUTION = "O96"
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["vsw", "sot"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS = [1, 2]


def load_ckpt(grid_resolution: str) -> str:
    assert grid_resolution in ["O96"], "Grid resolution must be O96"
    url = f"https://object-store.os-api.cci1.ecmwf.int/ml-tests/test-data/samples/training-course/inference-aifs-{grid_resolution}.ckpt"

    ckpt_file = Path("checkpoints") / f"aifs-global-{grid_resolution}.ckpt"

    # Create the output directory
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)

    # Download the checkpoint
    if not Path(ckpt_file).exists():
        process = subprocess.run(
            ["wget", url, "-O", ckpt_file],
            capture_output=True,
            text=True
        )
        print(process.stdout)
        print(process.stderr)
    else:
        print(f"Checkpoint already exists in {ckpt_file}")
    return str(ckpt_file)


def load_saved_state(file) -> dict:
    # Get the date from the filename: f"inputstate-{}-{date.strftime('%Y%m%d_%H')}.npz"
    date_str = file.stem.split("-")[2]
    date = datetime.datetime.strptime(date_str, "%Y%m%d_%H")

    with np.load(file, allow_pickle=False) as data:
        fields = {k: data[k] for k in data.files}

    state = {"date": date, "fields": fields}
    return state

def load_state(file):
    with np.load(file, allow_pickle=False) as data:
        fields = {k: data[k] for k in data.files}

    return {"fields": fields, "latitudes": fields.pop("latitudes"), "longitudes": fields.pop("longitudes")}

def save_state(state, outfile):
    np.savez(outfile, **state["fields"], longitudes = state["longitudes"], latitudes = state["latitudes"])

def get_open_data(param, date, resolution: str, levelist=[]) -> dict:
    fields = defaultdict(list)
    # Get the data for the current date and the previous date
    for date in [date - datetime.timedelta(hours=6), date]:
        data = ekd.from_source("ecmwf-open-data", date=date, param=param, levelist=levelist)
        for f in data:
            # Open data is between -180 and 180, we need to shift it to 0-360
            assert f.to_numpy().shape == (721, 1440)
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            # Interpolate the data to from 0.25 to grid
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": resolution})
            # Add the values to the list
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist else f.metadata("param")
            fields[name].append(values)

    # Create a single matrix for each parameter
    for param, values in fields.items():
        fields[param] = np.stack(values)

    return fields


def rename_keys(state: dict, mapping: dict) -> dict:
    for old_key, new_key in mapping.items():
        if old_key in state:
            state[new_key] = state.pop(old_key)
        else:
            LOGGER.warning(f"Key {old_key} not found in state, cannot rename to {new_key}")

    return state


def transform_GH_to_Z(fields: dict, levels: list[str]) -> dict:
    for level in levels:
        fields[f"z_{level}"] = fields.pop(f"gh_{level}") * 9.80665

    return fields


def load_opendata_state(resolution: str, date: str) -> dict:
    fields = {}
    fields.update(get_open_data(param=PARAM_SFC, date=date, resolution=resolution))
    #fields.update(get_open_data(param=PARAM_SOIL, date=date, resolution=resolution, levelist=SOIL_LEVELS))
    fields.update(get_open_data(param=PARAM_PL, date=date, resolution=resolution, levelist=LEVELS))

    fields = rename_keys(fields, {'sot_1': 'stl1', 'sot_2': 'stl2', 'vsw_1': 'swvl1', 'vsw_2': 'swvl2'})
    fields = transform_GH_to_Z(fields, LEVELS)

    return dict(date=date, fields=fields)


def save_state(state, outfile):
    np.savez(outfile, **state["fields"])


def load_input_state(date, resolution: str = GRID_RESOLUTION) -> dict:
    file = Path(f"inputstate-{resolution}-{date.strftime('%Y%m%d_%H')}.npz")
    if file.exists():
        input_state = load_saved_state(file)
        LOGGER.info(f"Input state loaded from file {file}")
    else:
        input_state = load_opendata_state(resolution=resolution, date=date)
        LOGGER.info(f"State created for date {date}")
        save_state(input_state, file)
        LOGGER.info(f"State saved to file {file}")

    return input_state


def haversine(coords: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """Compute haversine distance between multiple coordinates and a single point.

    Args:
        coords: (N, 2) tensor of [lat, lon] in degrees
        point: (2,) tensor of [lat, lon] in degrees

    Returns:
        (N,) tensor of distances in kilometers
    """
    coords_rad = torch.deg2rad(coords)
    point_rad = torch.deg2rad(point)

    lat1, lon1 = coords_rad[:, 0], coords_rad[:, 1]
    lat2, lon2 = point_rad[0], point_rad[1]

    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return R * c


def compute_sensitivities_statistics(sensitivities) -> pd.DataFrame:
    df = pd.DataFrame()
    for i, (varname, array) in enumerate(sensitivities["fields"].items()):
        var_type = "pl" if "_" in varname and varname.split("_")[1].isdigit() else "sfc"
        pl = varname.split("_")[-1] if var_type == "pl" else None
        name = varname.split("_")[0] if var_type == "pl" else varname
        df[2 * i] = (
            varname,
            var_type,
            name,
            pl,
            -6,
            np.sum(array[0] != 0) / array[0].size * 100,
            np.min(array[0]),
            np.max(array[0]),
        )
        df[2 * i + 1] = (
            varname,
            var_type,
            name,
            pl,
            0,
            np.sum(array[1] != 0) / array[1].size * 100,
            np.min(array[1]),
            np.max(array[1]),
        )
    df = df.T
    df.columns = ["var", "type", "varname", "pl", "hour", "num_nonzero", "min", "max"]
    df = df.astype({"num_nonzero": "float32", "min": "float32", "max": "float32"})
    df = df.set_index(["var", "type", "varname", "pl", "hour"])
    df = df.round({"num_nonzero": 2}).reset_index()
    return df

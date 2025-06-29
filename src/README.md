# :folder: `src` Directory Overview

This folder contains all core source code modules, organized into data utilities, models, and plotting tools.

| Subdirectory          | Description                           | Files                                                       |
|-----------------------|---------------------------------------|-------------------------------------------------------------|
| [`data`](./data)      | Data download, preparation, and utilities | - `clip_data.py`<br>- `data_download.py`<br>- `data_utils.py`<br>- `generate_synthetic_data.py` |
| [`models`](./models)  | Model configuration and VSH modules  | - `configuration.py`<br>- `vsh_model.py`                  |
| [`plot`](./plot)      | Plotting functions                   | - `plots.py`                                              |
| *(root)*              | Pickle and saving tools             | - `save_load_pkl.py`                                     |

---

## :speech_balloon: Notes

- **`data_download.py`**: Handles fetching data (e.g., Gaia queries).
- **`data_utils.py`**: Binning and preprocessing data.
- **`vsh_model.py`**: Core Vector Spherical Harmonics (VSH) implementation.
- **`plots.py`**: Contains all plotting functions for results visualization.
- **`save_load_pkl.py`**: For saving and loading model or intermediate data.

---

:white_check_mark: Click on each subdirectory link above to explore the individual files in detail!

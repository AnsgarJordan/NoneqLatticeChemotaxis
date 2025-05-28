import numpy as np
import json
import os
from datetime import datetime, timezone, timedelta

class ResultsSaver:
    def __init__(self, base_dir="results"):
        self.base_dir = base_dir
        self.folder_path = None

    def _convert_types(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Unserializable object {obj} of type {type(obj)}")

    def create_folder(self):
        gmt_plus_1 = timezone(timedelta(hours=1))
        now = datetime.now(gmt_plus_1)
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S%z")
        self.folder_path = os.path.join(self.base_dir, date_str)
        os.makedirs(self.folder_path, exist_ok=True)
        return self.folder_path

    def save_results(self, results, filename="simulation_results.npz"):
        if self.folder_path is None:
            self.create_folder()
        npz_path = os.path.join(self.folder_path, filename)
        np.savez_compressed(npz_path, **results)
        print(f"Results saved to {npz_path}")
        return npz_path

    def save_params(self, params, filename="simulation_params.json"):
        if self.folder_path is None:
            self.create_folder()
        json_path = os.path.join(self.folder_path, filename)
        with open(json_path, "w") as f:
            json.dump(params, f, default=self._convert_types, indent=4)
        print(f"Parameters saved to {json_path}")
        return json_path

    def save(self, results, params, results_filename="simulation_results.npz", params_filename="simulation_params.json"):
        self.create_folder()
        self.save_results(results, filename=results_filename)
        self.save_params(params, filename=params_filename)

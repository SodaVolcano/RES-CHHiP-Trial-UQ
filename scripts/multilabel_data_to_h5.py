"""
quick and dirty script to preprocess and dump multi-label dataset into a h5
"""

import sys

sys.path.append(".")
sys.path.append("..")
import os

import h5py
import toolz as tz
from tqdm import tqdm

from uncertainty.data.dicom import (
    _load_roi_mask,
    _load_rt_struct,
    _preprocess_mask,
    load_volume,
)

out_path = "temp.h5"
observers = ["DR", "JD", "JdL", "KL", "MAE", "MEB", "MK", "RR", "RTW", "SAB"]
organs = ["CTV", "Bladder", "Rectum"]
dicom_path = "./newcastle/"


paths = [os.path.join(dicom_path, folder) for folder in os.listdir(dicom_path)]

with h5py.File(out_path, "w") as hf:
    for i, path in tqdm(enumerate(paths), total=5):
        group = hf.create_group(str(i))
        # group.create_dataset("x", data=load_volume(path), compression="gzip")
        ys_group = group.create_group("y")

        rt_struct = _load_rt_struct(path)

        for organ in organs:
            masks = []
            for observer in observers:
                name = f"{organ}_{observer}"
                # ctv is sometimes called ctv_ct
                if name not in rt_struct.get_roi_names():
                    name = f"{organ}_CT_{observer}"

                mask = _load_roi_mask(rt_struct, name)
                if mask is None:
                    print(name, None, path)
                    continue
                # masks.append(list(_preprocess_mask([mask], path))[0][1])

            key = "prostate" if organ == "CTV" else organ.lower()
            # ys_group.create_dataset(key, data=masks, compression="gzip")

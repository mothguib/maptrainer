# -*- coding: utf-8 -*-

# `loaddl` for "load data loader"

from .. import data


def load_data_loader(dl_type: str = "BP") -> type:
    dl_name = dl_type + "DataLoader"

    return getattr(data, dl_name)

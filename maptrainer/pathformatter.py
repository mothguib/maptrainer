# -*- coding: utf-8 -*-

import time
import os
import re

from maptrainer import SAVES, MAPS


def generate_modelname(tpl,
                       nlayers: int = -1,
                       nhid: int = -1,
                       nagts=-1,
                       bptt: int = -1,
                       pre: bool = False,
                       arch: str = None):
    """
    Generates model name from parameters.

    :param tpl:
    :type tpl:
    :param nagts:
    :type nagts:
    :param nlayers:
    :type nlayers:
    :param nhid:
    :type nhid:

    :param bptt:
    :type bptt:
    :param arch: architecture of the model whose the name pattern is
    `<nlayers>-<nhid>`
    :type arch:
    :param pre: pre-model or not
    :type pre:
    :return:
    :rtype:
    """

    if arch is None:
        if nlayers != -1 and nhid != -1:
            arch = "{}-{}".format(nlayers, nhid)
        else:
            raise ValueError("`arch` or `nlayers` and `nhid` must be set.")

    model_sfx = "{}-{}".format(arch, bptt)

    if pre:
        return re.sub("--.", "--1", "pre-{}-{}".format(tpl, model_sfx))  #
        # Hack to guaranty that the RNN pre-models contain always a BPTT of
        # "--1". Indeed, RNN pre-models will always have a BPTT of -1,
        # by design
        # TODO: rebuild to take into account this
    else:
        if nagts != -1:
            return "{}-{}-{}". \
                format(tpl, nagts, model_sfx)
        else:
            return "{}-{}". \
                format(tpl, model_sfx)


def generate_model_file_name(_type: str,
                             _map: str,
                             nlayers: int,
                             nhid: int,
                             bptt: int,
                             pre: bool,
                             nagts: int,
                             epochs: int,
                             lr: float,
                             bsz: int,
                             clip: float,
                             dropout: float,
                             folds: int,
                             alr: bool,
                             timestamp: int = time.time()):
    """

    The file name pattern is
    <modelname>.<time_stamp>.<hyper-parameters>.<ext> as follows:
    the first dot separates the model's name from the time stamp, the
    second one separates the latter from the model's hyper-parameter,
    then the subsequent dots correspond to ones of the floating
    hyper-parameters and the last one separates the rest of the file
    name from the extension

    :param _map:
    :type _map:
    :param nlayers:
    :type nlayers:
    :param nhid:
    :type nhid:
    :param bptt:
    :type bptt:
    :param nagts:
    :type nagts:
    :param epochs:
    :type epochs:
    :param lr:
    :type lr:
    :param bsz:
    :type bsz:
    :param clip:
    :type clip:
    :param dropout:
    :type dropout:
    :param folds:
    :type folds:
    :param alr:
    :type alr:
    :param _type: type of file name: model, log or cost
    :type _type:
    :param pre: pre-model or not
    :type pre:
    :param timestamp:
    :type timestamp:
    :return:
    :rtype:
    """

    ext = {"model": "pt", "cost": "json", "log": "txt", "tb": "tb", "ac": "ac"}

    if alr:
        fn = "{}-{}.{}.{}.{}-{}-{}-{}-{}-alr.{}" \
            .format(_type,
                    generate_modelname(tpl=_map, nlayers=nlayers,
                                       nhid=nhid, bptt=bptt,
                                       pre=pre, nagts=nagts),
                    timestamp, epochs, lr, bsz, clip, dropout, folds,
                    ext[_type])
    else:
        fn = "{}-{}.{}.{}.{}-{}-{}-{}-{}.{}" \
            .format(_type,
                    generate_modelname(tpl=_map, nlayers=nlayers,
                                       nhid=nhid, bptt=bptt,
                                       pre=pre, nagts=nagts),
                    timestamp, epochs, lr, bsz, clip, dropout, folds,
                    ext[_type])

    return fn


def generate_model_dirname(_map: str, nlayers: int, nhid: int,
                           bptt: int,
                           pre: bool, nagts: int = -1):
    return generate_modelname(tpl=_map, nlayers=nlayers, nhid=nhid,
                              bptt=bptt, pre=pre, nagts=nagts) + '/'


def split_modelname(model: str):
    # Model's parameters: model's name is split to get its parameters
    mps = model.split('-')

    _map = mps[0]
    nlayers = int(mps[2])
    nhid = int(mps[3])
    bptt = -int(mps[5]) if mps[4] == '' else int(mps[4])

    return _map, nlayers, nhid, bptt


def generate_premodelname(model: str):
    """
    Returns the pre-model directory from the model name

    :param model: the model name
    :type model:
    :return:
    :rtype:
    """

    # Model's parameters: model's name is split to get its parameters
    _map, nlayers, nhid, bptt = split_modelname(model)
    pre = True

    return generate_modelname(tpl=_map, nlayers=nlayers, nhid=nhid,
                              bptt=bptt, pre=pre)


def generate_premodel_dir(model: str):
    """
    Returns the pre-model directory from the model name

    :param model:
    :type model:
    :return:
    :rtype:
    """

    # Model's parameters: model's name is split to get its parameters
    _map, nlayers, nhid, bptt = split_modelname(model)
    pre = True

    return generate_model_dirname(_map, nlayers, nhid, bptt, pre)


def generate_premodel_dirpath(map_modelsdirpath: str):
    """
    Generates the directory path of the pre-models for the current map's
    models from their directory path `map_modelsdirpath`

    :param map_modelsdirpath: directory path of the current type and
    variant model
    :type map_modelsdirpath:
    :return:
    :rtype:
    """

    # Replacement of multiple `/` with only one `/`
    map_modelsdirpath = re.sub("/+", '/', map_modelsdirpath)

    # Deletion of the last `/` if it exists
    map_modelsdirpath = map_modelsdirpath[:-1] \
        if map_modelsdirpath[-1] == '/' else map_modelsdirpath

    # Splited path
    sp = map_modelsdirpath.split('/')

    datasrc = sp[-1]

    return map_modelsdirpath.replace(datasrc, "pre")


def generate_save_profile_directory_path(log_rep_dir: str,
                                         model_type: str,
                                         model_variant: str,
                                         tpl: str,
                                         pre: bool,
                                         datasrc: str = 'hpcc_0.5',
                                         sspdp_suffix: str = '') -> str:
    """
    Generates the directory path for the current *save profile* {<log
    repository type>, <model type>, <model variant>, <map>,
    <data source>} where the save type corresponds to the log file type:
    `models`, `costs` or `logs`.

    :param tpl:
    :type tpl:
    :param log_rep_dir: repository directory for the current type of logs
    files to handle: `models`, `costs` or `logs`.
    :type log_rep_dir: str
    :param model_type:
    :param model_variant:
    :param pre:
    :type pre:

    :param datasrc: source whence data were drawn
    :type datasrc: str
    :param sspdp_suffix: further sub-directories to add in the path
    :type sspdp_suffix:
    :return:
    """

    # The save directory path for the current model type and variant
    dirpath = "{}/{}/{}/{}/{}/{}/".format(log_rep_dir, model_type,
                                          model_variant, tpl,
                                          ("pre/" if pre
                                           else datasrc),
                                          sspdp_suffix)

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    return dirpath


def generate_savefile_dirpath(type_: str,
                              tpl: str,
                              nlayers: int,
                              nhid: int,
                              bptt: int = -1,
                              pre: bool = False,
                              nagts: int = -1,
                              model_type: str = "RNN",
                              model_variant: str = "LSTM",
                              log_rep_dir: str = None,
                              modelname: str = None,
                              datasrc: str = None,
                              suffix: str = None):
    """
    Generates the directory path of the current save file of type `type`
    for the current model. There are three types of save file: models,
    costs and logs.

    :param type_:
    :type type_: str
    :param tpl:
    :type tpl: str
    :param nlayers:
    :type nlayers: int
    :param nhid:
    :type nhid: int
    :param bptt:
    :type bptt: int
    :param pre:
    Default: False
    :type pre: bool
    :param nagts:
    :type nagts: int
    :param model_type:
    :type model_type: str
    :param model_variant:
    :type model_variant: str
    :param log_rep_dir:
    :type log_rep_dir: str
    :param modelname:
    :type modelname: str
    :param datasrc: source whence data were drawn
    :type datasrc: str
    :param suffix:
    :type suffix:
    :return:
    :rtype:
    """

    if modelname is None:
        modelname = generate_modelname(tpl=tpl,
                                       nlayers=nlayers,
                                       nhid=nhid,
                                       bptt=bptt, pre=pre,
                                       nagts=nagts)

    if log_rep_dir is None:
        plurals = {"model": "models", "cost": "costs", "log": "logs"}
        log_rep_dir = SAVES + plurals[type_]

    if datasrc is None:
        datasrc = "hcc_0.2"

    if suffix is None:
        suffix = ''

    # `spdp` abbreviation of "variant type save directory path": directory
    #  of the save file for the current type and variant of model
    spdp = generate_save_profile_directory_path(log_rep_dir=log_rep_dir,
                                                model_type=model_type,
                                                model_variant=model_variant,
                                                tpl=tpl,
                                                pre=pre,
                                                datasrc=datasrc)

    return "{}/{}/{}/".format(spdp, modelname, suffix)


def generate_savefilepath(
        type_: str,
        tpl: str,
        nlayers: int,
        nhid: int,
        bptt: int = -1,
        pre: bool = False,
        nagts: int = -1,
        epochs: int = 1000,
        lr: float = 0.1,
        bsz: int = -1,
        clip: int = -1,
        dropout: float = 0.0,
        folds: int = 1,
        model_type: str = "SMRNN",
        model_variant: str = "LSTM",
        log_rep_dir: str = None,
        alr: bool = False,
        timestamp: int = time.time(),
        datasrc: str = ''):
    """
    Generates the path of the current save file of `_type` type for
    the current model. There are three types of save file: the models,
    costs and logs.

    :param tpl:
    :type tpl:
    :param type_:
    :type type_:
    :param nlayers:
    :type nlayers: int
    :param nhid:
    :type nhid: int
    :param bptt:
    :type bptt: int
    :param pre:
    :type pre: bool
    :param nagts:
    :type nagts: int
    :param epochs:
    :type epochs: int
    :param lr:
    :type lr: float
    :param bsz:
    :type bsz: int
    :param clip:
    :type clip: float
    :param dropout:
    :type dropout: float
    :param folds:
    :type folds: int
    :param alr:
    :type alr: bool
    :param model_type:
    :type model_type: str
    :param model_variant:
    :type model_variant: str
    :param log_rep_dir:
    :type log_rep_dir: str
    :param timestamp:
    :type timestamp: int
    :param datasrc: source whence data were drawn
    :type datasrc: str
    :return:
    :rtype:


    """

    if log_rep_dir is None:
        plurals = {"model": "models", "cost": "costs", "log": "logs",
                   "tb": "tbs", "ac": "accuracies"}
        log_rep_dir = SAVES + plurals[type_]

    # `spdp` abbreviation of "variant type save directory path": directory
    #  of the save file for the current type and variant of model
    spdp = generate_save_profile_directory_path(log_rep_dir=log_rep_dir,
                                                model_type=model_type,
                                                model_variant=model_variant,
                                                tpl=tpl,
                                                pre=pre,
                                                datasrc=datasrc)

    return "{}/{}/{}".format(spdp,
                             generate_model_dirname(tpl, nlayers,
                                                    nhid, bptt,
                                                    pre, nagts),
                             generate_model_file_name(type_, tpl,
                                                      nlayers, nhid,
                                                      bptt, pre, nagts,
                                                      epochs, lr, bsz,
                                                      clip, dropout,
                                                      folds, alr,
                                                      timestamp)
                             )


def entire_model_path(path: str):
    if ".entire" not in path:
        path = path.replace(".pt", ".entire.pt")

    if "/entire/" not in path:
        # Replacement of multiple `/` with only one `/`
        path = re.sub("/+", '/', path)

        # Splited path
        sp = path.split('/')

        sp.insert(-1, "entire")

        return '/'.join(sp)

    return path


def topology_path(tn: str):
    """

    :param tn: topology name
    :type tn: str
    :return:
    """

    return MAPS + (tn if tn.endswith(".bin.json") else tn + ".bin.json")

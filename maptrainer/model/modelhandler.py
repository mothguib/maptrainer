# -*- coding: utf-8 -*-

import os
import torch

import maptrainer.pathformatter as pf
import maptrainer.misc as misc
from .. import model as modelpckg


def save_model(model: torch.nn.Module, path: str, entire: bool = True):
    """

    :param model: the model object
    :param path: the model's path
    :param entire: saving the entire model
    :return:
    """

    torch.save(model.state_dict(), path)

    if entire:
        path = pf.entire_model_path(path)
        dirpath = os.path.dirname(path)

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        with open(path, "wb") as s:
            torch.save(model, s)


def load_model_state_from_path(model: torch.nn.Module, path: str,
                               cuda: bool = False):
    with open(path, "rb") as s:
        if cuda:
            # Loading on GPU if trained with CUDA
            model.load_state_dict(torch.load(s))
        else:
            # Loading on CPU if trained with CUDA
            model.load_state_dict(
                torch.load(s, map_location=lambda storage, loc: storage)
            )

    return model


def load_model(model_type: str,
               model_variant: str,
               modelname: str,
               ninputs: int,
               nlayers: int,
               nhid: int,
               model_dirpath: str,
               dropout: float = 0.0,
               noutputs: int = -1,
               pre: bool = False,
               check_exists: bool = False,
               check_pre_exists: bool = False,
               graph=None) -> torch.nn.Module:
    """

    :param noutputs:
    :type noutputs:
    :param check_pre_exists:
    :type check_pre_exists:
    :param model_type:
    :type model_type:
    :param model_variant:
    :type model_variant:
    :param modelname:
    :type modelname:
    :param ninputs:
    :type ninputs:
    :param nhid:
    :type nhid:
    :param nlayers:
    :type nlayers:
    :param dropout:
    :type dropout:
    :param pre: indicates if it is a pre-model or not
    :type pre: bool
    :param model_dirpath: directory path of the save files for the current
    model type and variant, the default pattern is:
    models/<type>/<variant>/<map> for basic models
    and models/<type>/<variant>/<pre>/<map> for pre-models
    :type model_dirpath:
    :param check_exists: checking the existence of a previous version of the
    model
    :type check_exists:
    :param check_pre_exists: checking, whether needed, the existence of a
    previous version of the pre-model
    :type check_pre_exists:
    :param graph:
    :type graph:
    :return:
    :rtype:
    """

    classname = model_type + "Model"
    model_class = getattr(modelpckg, classname)  # if it is not in the same
    # module
    # modelclass = globals()[classname]  # if it is in the same module

    model = model_class(ninputs=ninputs, nhid=nhid, nlayers=nlayers,
                        noutputs=noutputs, variant=model_variant,
                        dropout=dropout, graph=graph)
    if check_exists:
        if model_exists(modelname, model_dirpath):
            # Path of the latest version of the current model
            lv_model_dirpath = lv_model_path(modelname, model_dirpath)

            print("* `modelhandler.load_model`: model loaded: ",
                  lv_model_dirpath,
                  '\n')

            return load_model_state_from_path(model=model,
                                              path=lv_model_dirpath)

    if check_pre_exists:
        # If it is not a pre-model, it is checked whether it exists an
        # available pre-model
        if not pre:
            premodelname = pf.generate_premodelname(modelname)
            premodel_dirpath = pf \
                .generate_premodel_dirpath(model_dirpath)

            # If a pre-model exists
            if model_exists(premodelname, premodel_dirpath):
                print("* `modelhandler.load_model`: a pre-model exists in ",
                      premodel_dirpath, '\n')

                # If the pre-model exist, its latest saved version is returned
                return load_model(model_type=model_type,
                                  model_variant=model_variant,
                                  modelname=premodelname, ninputs=ninputs,
                                  nhid=nhid, nlayers=nlayers, dropout=dropout,
                                  pre=False, model_dirpath=premodel_dirpath,
                                  check_exists=True, noutputs=noutputs)

    # print("* `model.load_model`: the model", modelname, "does not exist",
    #      '\n')

    return model


def model_exists(model: str, model_dirpath: str):
    """

    :param model: model's name
    :type model: str
    :param model_dirpath: directory path of models for the current type and
    variant
    :type model_dirpath:
    :param pre: boolean to check if the model existence is carried out for a
    standard model or a pre-model
    :type pre:
    :return:
    :rtype:
    """

    dirs = os.listdir(model_dirpath)

    for d in dirs:
        if d == model:
            if len(os.listdir(model_dirpath + '/' + d)) > 0:
                print("* `modelhandler.model_exists`: '", d,
                      "' is present in '",
                      model_dirpath, "'", '\n')

                return True
    return False


def lv_model_path(modelname: str, model_dirpath: str) -> str:
    """
    Getting the path of the latest version of the model `modelname`

    :param modelname:
    :type modelname:
    :param model_dirpath: directory path of the save files for the current
    model type and variant, default pattern is: models/<type>/<variant>/<map>
    for basic models and models/<type>/<variant>/<pre>/<map> for pre-models
    :type model_dirpath:
    :return: returns the path of the latest version of the model `modelname`
    :rtype: str
    """

    # Path of the required model's directory whose the name is `modelname`
    model_dirpath = model_dirpath + '/' + modelname + '/'

    # Path of the latest version of the current model
    lvm_path = misc.get_latest_file(model_dirpath)

    return lvm_path

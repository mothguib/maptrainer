# -*- coding: utf-8 -*-

import json
import time
import torch

import numpy as np
import os

import maptrainer.pathformatter as pf


def grad_norm(parameters, norm_type=2):
    """Computes gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will
        have gradients normalized max_norm (float or int): max norm of the
        gradients norm_type (float or int): type of the used p-norm. Can be
        ``'inf'`` for infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm


def get_timestamp(fn):
    """
    Return the file `fn`'s timestamp

    :param fn:
    :type fn:
    :return:
    :rtype:
    """
    return int(fn.split('/')[-1].split('.')[1])


def get_latest_file(dirpath):
    """

    :param dirpath: Path of directory where find the latest file
    :return:
    """
    return \
        sorted(
            list(
                filter(os.path.isfile,
                       [dirpath + f for f in os.listdir(dirpath)])
            ),
            key=lambda fn: get_timestamp(fn)
        )[-1]


def init_saves(args):
    model_path = pf.generate_savefilepath(type_="model",
                                          tpl=args.map,
                                          nlayers=args.nlayers,
                                          nhid=args.nhid,
                                          bptt=args.bptt,
                                          pre=args.pre,
                                          nagts=args.nagts,
                                          epochs=args.epochs,
                                          lr=args.learning_rate,
                                          bsz=args.bsz,
                                          clip=args.clip,
                                          dropout=args.dropout,
                                          folds=args.folds,
                                          alr=args.alr,
                                          model_type=args.model_type,
                                          model_variant=args.model_variant,
                                          log_rep_dir=args.model_save,
                                          timestamp=args.timestamp,
                                          datasrc=args.datasrc)

    print("Path of the model:\n", model_path, "\n")

    directory = os.path.dirname(model_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    # File name of costs for the current model
    cost_path = pf.generate_savefilepath(type_="cost",
                                         tpl=args.map,
                                         nlayers=args.nlayers,
                                         nhid=args.nhid,
                                         bptt=args.bptt,
                                         pre=args.pre,
                                         nagts=args.nagts,
                                         epochs=args.epochs,
                                         lr=args.learning_rate,
                                         bsz=args.bsz,
                                         clip=args.clip,
                                         dropout=args.dropout,
                                         folds=args.folds,
                                         alr=args.alr,
                                         model_type=args.model_type,
                                         model_variant=args.model_variant,
                                         log_rep_dir=args.cost_save,
                                         timestamp=args.timestamp,
                                         datasrc=args.datasrc)

    print("Path of the costs' file:\n", cost_path, "\n")

    directory = os.path.dirname(cost_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Log's file name
    log_path = pf.generate_savefilepath(type_="log",
                                        tpl=args.map,
                                        nlayers=args.nlayers,
                                        nhid=args.nhid,
                                        bptt=args.bptt,
                                        pre=args.pre,
                                        nagts=args.nagts,
                                        epochs=args.epochs,
                                        lr=args.learning_rate,
                                        bsz=args.bsz,
                                        clip=args.clip,
                                        dropout=args.dropout,
                                        folds=args.folds,
                                        alr=args.alr,
                                        model_type=args.model_type,
                                        model_variant=args.model_variant,
                                        log_rep_dir=args.log_save,
                                        timestamp=args.timestamp,
                                        datasrc=args.datasrc)

    print("Path of the log file:\n", log_path, "\n")

    directory = os.path.dirname(log_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    # TensorBoard log's file name
    tb_path = pf.generate_savefilepath(type_="tb",
                                       tpl=args.map,
                                       nlayers=args.nlayers,
                                       nhid=args.nhid,
                                       bptt=args.bptt,
                                       pre=args.pre,
                                       nagts=args.nagts,
                                       epochs=args.epochs,
                                       lr=args.learning_rate,
                                       bsz=args.bsz,
                                       clip=args.clip,
                                       dropout=args.dropout,
                                       folds=args.folds,
                                       alr=args.alr,
                                       model_type=args.model_type,
                                       model_variant=args.model_variant,
                                       log_rep_dir=args.tb_save,
                                       timestamp=args.timestamp,
                                       datasrc=args.datasrc)

    print("Path of the TensorBoardX log file:\n", tb_path, "\n")

    directory = os.path.dirname(tb_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Accuracy's file name
    ac_path = pf.generate_savefilepath(type_="ac",
                                       tpl=args.map,
                                       nlayers=args.nlayers,
                                       nhid=args.nhid,
                                       bptt=args.bptt,
                                       pre=args.pre,
                                       nagts=args.nagts,
                                       epochs=args.epochs,
                                       lr=args.learning_rate,
                                       bsz=args.bsz,
                                       clip=args.clip,
                                       dropout=args.dropout,
                                       folds=args.folds,
                                       alr=args.alr,
                                       model_type=args.model_type,
                                       model_variant=args.model_variant,
                                       log_rep_dir=args.ac_save,
                                       timestamp=args.timestamp,
                                       datasrc=args.datasrc)

    print("Path of the accuracy file:\n", ac_path, "\n")

    directory = os.path.dirname(ac_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    return model_path, log_path, cost_path, tb_path, ac_path


def cuda(obj, cuda: bool = False, v: bool = False):
    """

    :param obj: the obj to wrap with cuda if activated. This obj shall be a
    tensor or a Pytorch module
    :type obj:
    :param cuda: cuda activation
    :type cuda:
    :param v: verbose mode
    :type v: bool
    :return:
    :rtype:
    """

    if cuda:
        start = time.time()
        obj = obj.cuda()
        end = time.time()
        if v:
            print("Loading time of ", obj, " on GPU: ", end - start, '\n')
    else:
        obj = obj.cpu()

    return obj
    # return obj.cuda() if ca else obj.cpu()


def get_end_costs(maps: list,
                  nagts: list,
                  archs: list,
                  bptts: list,
                  pathcosts: str,
                  modeltype: str,
                  modelvariant: str,
                  pre: bool = False,
                  datasrc: str = '') -> dict:
    """
    Returns the first and latest cost values for each combination of
    parameters passed to the function.

    :param pathcosts:
    :type pathcosts:
    :param datasrc:
    :type datasrc:
    :param maps:
    :type maps: list
    :param nagts:
    :type nagts: list
    :param archs:
    :type archs: list
    :param bptts:
    :type bptts: list
    :param modeltype:
    :type modeltype: str
    :param modelvariant:
    :type modelvariant: str
    :param pre:
    :type pre: bool
    :param datasrc: source whence data were drawn
    :type datasrc: str
    :return:
    :rtype:
    """

    if pre:
        nagts = ["pre"] + nagts

    costs = {}

    for m in maps:
        costs[m] = {}
        for n in nagts:
            costs[m][n] = {}
            for i, a in enumerate(archs):
                costs[m][n][a] = {}

                pre = True if n == "pre" else False
                modelname = pf.generate_modelname(m,
                                                  arch=a,
                                                  bptt=bptts[i],
                                                  pre=pre,
                                                  nagts=n)

                nlayers = a.split('-')[0]
                nhid = a.split('-')[1]

                dirpath = pf \
                    .generate_savefile_dirpath(type_="cost",
                                               tpl=m,
                                               pre=pre,
                                               nlayers=nlayers,
                                               nhid=nhid,
                                               model_type=modeltype,
                                               model_variant=modelvariant,
                                               modelname=modelname,
                                               log_rep_dir=pathcosts,
                                               datasrc=datasrc)

                if os.path.exists(dirpath):
                    # Files of the cost for the current model
                    paths_filescost = \
                        sorted(
                            list(
                                filter(os.path.isfile,
                                       ["{}/{}".format(dirpath, f)
                                        for f in os.listdir(dirpath)])
                            )
                        )

                    # Oldest file path of the cost evolution for the
                    # current model
                    initial_pathcost = paths_filescost[0]

                    with open(initial_pathcost) as s:
                        # Initial table of costs
                        ti = json.load(s)

                    # Latest file path of the cost evolution for the
                    # current model
                    latest_pathcost = paths_filescost[-1]

                    with open(latest_pathcost) as s:
                        # Latest table of costs
                        tl = json.load(s)

                    # Initial validation cost
                    f_val_cost = ti[1][0]
                    # Latest pre validation cost
                    l_val_cost = tl[1][-1]

                    costs[m][n][a] = \
                        {"initial": f_val_cost, "latest": l_val_cost}

                    # print(f_val_cost, l_val_cost)

    return costs

def get_end_costs_suffixes(maps: list,
                  nagts: list,
                  archs: list,
                  bptts: list,
                  pathcosts: str,
                  modeltype: str,
                  modelvariant: str,
                  pre: bool = False,
                  datasrc: str = '',
                  suffixes: list = None) -> dict:
    """
    Returns the first and latest cost values for each combination of
    parameters passed to the function.

    :param pathcosts:
    :type pathcosts:
    :param datasrc:
    :type datasrc:
    :param maps:
    :type maps: list
    :param nagts:
    :type nagts: list
    :param archs:
    :type archs: list
    :param bptts:
    :type bptts: list
    :param modeltype:
    :type modeltype: str
    :param modelvariant:
    :type modelvariant: str
    :param pre:
    :type pre: bool
    :param datasrc: source whence data were drawn
    :type datasrc: str
    :param suffixes:
    :type suffixes:
    :return:
    :rtype:
    """

    if suffixes is None:
        suffixes = []

    if pre:
        nagts = ["pre"] + nagts

    costs = {}

    for m in maps:
        costs[m] = {}
        for n in nagts:
            costs[m][n] = {}
            for i, a in enumerate(archs):
                costs[m][n][a] = {}

                for suffix in suffixes:
                    pre = True if n == "pre" else False
                    modelname = pf.generate_modelname(m,
                                                      arch=a,
                                                      bptt=bptts[i],
                                                      pre=pre,
                                                      nagts=n)

                    nlayers = a.split('-')[0]
                    nhid = a.split('-')[1]

                    dirpath = pf \
                        .generate_savefile_dirpath(type_="cost",
                                                   tpl=m,
                                                   pre=pre,
                                                   nlayers=nlayers,
                                                   nhid=nhid,
                                                   model_type=modeltype,
                                                   model_variant=modelvariant,
                                                   modelname=modelname,
                                                   log_rep_dir=pathcosts,
                                                   datasrc=datasrc,
                                                   suffix=suffix)

                    if os.path.exists(dirpath):
                        # Files of the cost for the current model
                        paths_filescost = \
                            sorted(
                                list(
                                    filter(os.path.isfile,
                                           ["{}/{}".format(dirpath, f)
                                            for f in os.listdir(dirpath)])
                                )
                            )

                        # Oldest file path of the cost evolution for the
                        # current model
                        initial_pathcost = paths_filescost[0]

                        with open(initial_pathcost) as s:
                            # Initial table of costs
                            ti = json.load(s)

                        # Latest file path of the cost evolution for the
                        # current model
                        latest_pathcost = paths_filescost[-1]

                        with open(latest_pathcost) as s:
                            # Latest table of costs
                            tl = json.load(s)

                        # Initial validation cost
                        f_val_cost = ti[1][0]
                        # Latest pre validation cost
                        l_val_cost = tl[1][-1]

                        costs[m][n][a][suffix] = \
                            {"initial": f_val_cost, "latest": l_val_cost}

                        # print(f_val_cost, l_val_cost)

    return costs


def neighbours(v, t) -> list:
    """
    Returns the neighbours of `v` in `t`.

    :param v:
    :param t: the adjacent matrix representing the topology
    :type t: iterable
    :return:
    """

    t = np.array(t, dtype=np.uint8)

    return np.where(t[v] > 0)[0].tolist()


def save_progress(writer,
                  model,
                  tr_cost,
                  val_cost,
                  accuracy: float,
                  iteration: int):

    if model.variant == "LSTM":
        save_progress_lstm(writer, model, iteration)
    else:
        for nm, p in model.named_parameters():
            writer.add_histogram(nm, p.detach().cpu().numpy(), iteration)

    writer.add_scalars("costs", {'tr': tr_cost}, iteration)
    writer.add_scalars("costs", {'val': val_cost}, iteration)
    writer.add_scalar("accuracy", accuracy, iteration)


def save_progress_lstm(writer, model, iteration: int):
    nps = model.state_dict()

    nhid = model.nhid

    for l in range(model.nlayers):
        weights_ih_i = \
            nps["rnn.weight_ih_l" + str(l)][0:nhid]  # $W^{ix}$: weight
        # `ih` of the gate `i_t`

        weights_ih_f = \
            nps["rnn.weight_ih_l" + str(l)][nhid:2 * nhid]  # $W^{fx}$: weight
        # `ih` of the gate `f_t`

        weights_ih_g = \
            nps["rnn.weight_ih_l" + str(l)][-2 * nhid:-nhid]
        # $W^{gx}$: weight `ih` of the gate `g_t` (`c_t`)

        weights_ih_o = \
            nps["rnn.weight_ih_l" + str(l)][-nhid:]  # $W^{ox}$: weight
        # `ih` of the gate `o_t`

        weights_hh_i = \
            nps["rnn.weight_hh_l" + str(l)][0:nhid]  # $W^{ih}$: weight
        # `ih` of the gate `i_t`

        weights_hh_f = \
            nps["rnn.weight_hh_l" + str(l)][nhid:2 * nhid]  # $W^{fh}$: weight
        # `ih` of the gate `f_t`

        weights_hh_g = \
            nps["rnn.weight_hh_l" + str(l)][-2 * nhid:-nhid]
        # $W^{gh}$: weight `hh` of the gate `g_t` (`c_t`)

        weights_hh_o = \
            nps["rnn.weight_hh_l" + str(l)][-nhid:]
        # $W^{oh}$: weight `hh` of the gate `o_t`

        biases_ih_i = \
            nps["rnn.bias_ih_l" + str(l)][0:nhid]  # $b^{ix}$: bias
        # `ih` of the gate `i_t`

        biases_ih_f = \
            nps["rnn.bias_ih_l" + str(l)][nhid:2 * nhid]  # $b^{fx}$: bias
        # `ih` of the gate `f_t`

        biases_ih_g = \
            nps["rnn.bias_ih_l" + str(l)][-2 * nhid:-nhid]
        # $b^{gx}$: bias `ih` of the gate `g_t` (`c_t`)

        biases_ih_o = \
            nps["rnn.bias_ih_l" + str(l)][-nhid:]  # $b^{ox}$: bias
        # `ih` of the gate `o_t`

        biases_hh_i = \
            nps["rnn.bias_hh_l" + str(l)][0:nhid]  # $b^{ih}$: bias
        # `ih` of the gate `i_t`

        biases_hh_f = \
            nps["rnn.bias_hh_l" + str(l)][nhid:2 * nhid]  # $b^{fh}$: bias
        # `ih` of the gate `f_t`

        biases_hh_g = \
            nps["rnn.bias_hh_l" + str(l)][-2 * nhid:-nhid]
        # $b^{gh}$: bias `hh` of the gate `g_t` (`c_t`)

        biases_hh_o = \
            nps["rnn.bias_hh_l" + str(l)][-nhid:]
        # $b^{oh}$: bias `hh` of the gate `o_t`

        writer.add_histogram("rnn.weights_ih_i_l" + str(l),
                             weights_ih_i.detach().cpu().numpy(), iteration)
        writer.add_histogram("rnn.weights_ih_f_l" + str(l),
                             weights_ih_f.detach().cpu().numpy(), iteration)
        writer.add_histogram("rnn.weights_ih_g_l" + str(l),
                             weights_ih_g.detach().cpu().numpy(), iteration)
        writer.add_histogram("rnn.weights_ih_o_l" + str(l),
                             weights_ih_o.detach().cpu().numpy(), iteration)
        writer.add_histogram("rnn.weights_hh_i_l" + str(l),
                             weights_hh_i.detach().cpu().numpy(), iteration)
        writer.add_histogram("rnn.weights_hh_f_l" + str(l),
                             weights_hh_f.detach().cpu().numpy(), iteration)
        writer.add_histogram("rnn.weights_hh_g_l" + str(l),
                             weights_hh_g.detach().cpu().numpy(), iteration)
        writer.add_histogram("rnn.weights_hh_o_l" + str(l),
                             weights_hh_o.detach().cpu().numpy(), iteration)

        writer.add_histogram("rnn.biases_ih_i_l" + str(l),
                             biases_ih_i.detach().cpu().numpy(), iteration)
        writer.add_histogram("rnn.biases_ih_f_l" + str(l),
                             biases_ih_f.detach().cpu().numpy(), iteration)
        writer.add_histogram("rnn.biases_ih_g_l" + str(l),
                             biases_ih_g.detach().cpu().numpy(), iteration)
        writer.add_histogram("rnn.biases_ih_o_l" + str(l),
                             biases_ih_o.detach().cpu().numpy(), iteration)
        writer.add_histogram("rnn.biases_hh_i_l" + str(l),
                             biases_hh_i.detach().cpu().numpy(), iteration)
        writer.add_histogram("rnn.biases_hh_f_l" + str(l),
                             biases_hh_f.detach().cpu().numpy(), iteration)
        writer.add_histogram("rnn.biases_hh_g_l" + str(l),
                             biases_hh_g.detach().cpu().numpy(), iteration)
        writer.add_histogram("rnn.biases_hh_o_l" + str(l),
                             biases_hh_o.detach().cpu().numpy(), iteration)

    linearise_weight = nps["linearise.weight"]

    linearise_bias = nps["linearise.bias"]

    writer.add_histogram("linearise.weight",
                         linearise_weight.detach().cpu().numpy(), iteration)

    writer.add_histogram("linearise.bias",
                         linearise_bias.detach().cpu().numpy(), iteration)


# TODO: to delete
def test_data(train_ddata, train_tdata, val_ddata, val_tdata):
    s0 = val_tdata.size(0)
    s = val_tdata.size(0) * val_tdata.size(1)

    print("s: ", s)
    print("s0: ", s0)
    nb_val_tdata = torch.sum(val_tdata)

    print(torch.sum(train_tdata[:15]), torch.sum(
        torch.max(train_tdata[:15], 2)[1] == torch.max(train_tdata[15:30],
                                                       2)[1]))
    print(torch.sum(train_tdata[15:30]),
          torch.sum(torch.max(train_tdata[15:30],
                              2)[1] == torch.max(
              train_tdata[30:45], 2)[1]))
    print(torch.sum(torch.max(train_tdata[30:45], 2)[1] == torch.max(
        train_tdata[45:60], 2)[1]))
    print(torch.sum(torch.max(train_tdata[45:60], 2)[1] == torch.max(
        train_tdata[60:75], 2)[1]))

    print(torch.sum(torch.max(train_tdata[:s0], 2)[1] ==
                    torch.max(train_tdata[s0:2 * s0], 2)[1]))

    print(torch.sum(torch.max(train_tdata[:s0], 2)[1] ==
                    torch.max(train_tdata[:s0], 2)[1]))

    print(torch.sum(torch.max(val_tdata, 2)[1] ==
                    torch.max(train_tdata[:s0], 2)[1]))
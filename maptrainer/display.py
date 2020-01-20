# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

from maptrainer import misc
import maptrainer.pathformatter as pf
from maptrainer.argsparser import parse_args

plt.rcdefaults()

LINESCLRS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
LINESMRKS = ['o', 's', 'P', 'v', '^', '<', '>', '1', '2', '3', '4', '8']
LINESTYLES = ['-', ':']
FONTSIZE = 25
LGDFONTSIZE = 21.5
TITLEFONTSIZE = 22

CRITERIA = ["train", "valid"]


def plot_cost(_map: str,
              nagts: int,
              nlayers: int,
              nhid: int,
              bptt: int = -1,
              pre: bool = False,
              **kwargs):
    """

    :param _map:
    :type _map:
    :param nagts:
    :type nagts:
    :param nlayers:
    :type nlayers:
    :param nhid:
    :type nhid:
    :param bptt:
    :type bptt:
    :param pre:
    :type pre:
    :return:
    """

    args = parse_args()

    pathcosts = kwargs.pop("pathcosts", args.cost_save)
    costpath = kwargs.pop("pathcost", '')

    if costpath == '':
        modelname = pf.generate_modelname(tpl=_map,
                                          nlayers=nlayers,
                                          nhid=nhid, bptt=bptt,
                                          pre=pre, nagts=nagts)
        cost_dirpath = "{}/{}/{}/".format(pathcosts,
                                          "pre" if pre else '',
                                          modelname)
        costpath = misc.get_latest_file(cost_dirpath)

    # Loading of criteria to plot: ctrs
    with open(costpath, 'r') as s:
        ctrs = json.load(s)

    upperbound = kwargs.pop("upperbound", len(ctrs[0]))

    handles = []

    for i in range(len(ctrs)):
        crtr_line = lines.Line2D([], [], color=LINESCLRS[i], label=CRITERIA[i],
                                 linestyle='-')
        handles += [crtr_line]
        i += 1

    # Setting the axes' label font size
    # ax.tick_params(labelsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    for i, c in enumerate(ctrs):
        plt.plot(range(upperbound), np.exp(c[:upperbound]), color=LINESCLRS[i],
                 markersize=12, linestyle='-')

    plt.legend(handles=handles, prop={'size': LGDFONTSIZE})

    plt.show()


def plot_end_costs(maps: list,
                   nagts: list,
                   archs: list,
                   bptts: list,
                   pathcosts: str,
                   modeltype: str,
                   modelvariant: str,
                   pre: bool = False,
                   datasrc: str = ''):
    """

    :param maps:
    :type maps: list
    :param nagts:
    :type nagts: list
    :param archs:
    :type archs: list
    :param bptts:
    :type bptts: list
    :param modeltype:
    :type modeltype:
    :param modelvariant:
    :type modelvariant:
    :param pathcosts:
    :type pathcosts:
    :param pre:
    :type pre:
    :param datasrc: source whence data were drawn
    :type datasrc: str
    :return:
    :rtype:
    """

    # Array of legends for each bar
    legends = []
    # Array of the top stacked bars' values: extra value with respect to the
    # latest one
    extras = []
    # Array of the bottom stacked bars' values: the latest costs
    latests = []

    if pre:
        nagts = ["pre"] + nagts

    # Dictionary of costs
    c_dic = misc.get_end_costs(maps=maps, nagts=nagts, archs=archs,
                               bptts=bptts, modeltype=modeltype,
                               modelvariant=modelvariant,
                               pathcosts=pathcosts, datasrc=datasrc)
    for m in maps:
        for n in nagts:
            for a in archs:
                # The current architecture's latest cost
                latest = c_dic[m][n][a]["latest"]
                latests += [latest]
                # Difference between the initial costs and the latest
                # corresponding to the quantity needed to plot a stacked
                # bar chart
                extras += [c_dic[m][n][a]["initial"] - latest]

                legends += ["{} - {} - {}".format(m, n, a)
                            if len(maps) > 1
                            else "{} - {}".format(n, a)]

    ids = np.arange(len(legends))

    ax = plt.gca()  # Get the current `matplotlib.axes.Axes` instance on
    # the current figure matching the given keyword args, or create one.

    # Setting the axes' label font size
    # ax.tick_params(labelsize=FONTSIZE)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=FONTSIZE)
    plt.xticks(ids, legends, rotation=70)
    # ax.set_xticks(x, legends, rotation=70)

    rects1 = plt.bar(ids, extras, align='center', alpha=0.5, bottom=latests,
                     color="blue")
    rects2 = plt.bar(ids, latests, align='center', alpha=0.5, color="red")

    plt.ylabel("Cost", fontsize=FONTSIZE)

    plt.title("Validation costs at the begining and the end of training for "
              "{}".format(', '.join(maps)), fontsize=FONTSIZE)

    plt.legend((rects1[0], rects2[0]), ('Initial cost', 'Final cost'),
               fontsize=LGDFONTSIZE)

    def autolabel(rects1, rects2):
        """
        Attach a text label above each bar displaying its height
        """
        for i, rect in enumerate(rects1):
            height = rect.get_height() + rects2[i].get_height()
            h_factor = 1. if rect.get_height() - rects2[i].get_height() > 1 \
                else 1.05 if rect.get_height() - rects2[i].get_height() > 0.2 \
                else 1.1
            ax.text(rect.get_x() + rect.get_width() / 2., h_factor * height,
                    format_value(height),
                    ha='center', va='bottom', fontsize=FONTSIZE)

        for rect in rects2:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1. * height,
                    format_value(height),
                    ha='center', va='bottom', fontsize=FONTSIZE)

    def format_value(value):
        return "%2.1f" % value

    autolabel(rects1, rects2)

    plt.show()


def plot_end_costs_suffixes(maps: list,
                            nagts: list,
                            archs: list,
                            bptts: list,
                            pathcosts: str,
                            modeltype: str,
                            modelvariant: str,
                            pre: bool = False,
                            datasrc: str = ''):
    """

    :param maps:
    :type maps: list
    :param nagts:
    :type nagts: list
    :param archs:
    :type archs: list
    :param bptts:
    :type bptts: list
    :param modeltype:
    :type modeltype:
    :param modelvariant:
    :type modelvariant:
    :param pathcosts:
    :type pathcosts:
    :param pre:
    :type pre:
    :param datasrc: source whence data were drawn
    :type datasrc: str
    :rtype:
    """

    # Array of legends for each bar
    legends = []
    # Array of the top stacked bars' values: extra value with respect to the
    # latest one
    extras = []
    # Array of the bottom stacked bars' values: the latest costs
    latests = []

    # Suffixes
    suffixes = ["SGD-pre", "Adagrad-pre"]

    # Suffix abbreviations
    suffix_abrs = {"SGD-pre": "sp", "Adagrad-pre": "ap"}

    if pre:
        nagts = ["pre"] + nagts

    # Dictionary of costs
    c_dic = misc.get_end_costs_suffixes(maps=maps, nagts=nagts, archs=archs,
                                        bptts=bptts, modeltype=modeltype,
                                        modelvariant=modelvariant,
                                        pathcosts=pathcosts, datasrc=datasrc,
                                        suffixes=suffixes)

    for suffix in suffixes:
        for m in maps:
            for n in nagts:
                for a in archs:
                    # The current architecture's latest cost
                    latest = c_dic[m][n][a][suffix]["latest"]
                    latests += [latest]
                    # Difference between the initial costs and the latest
                    # corresponding to the quantity needed to plot a stacked
                    # bar chart
                    extras += [c_dic[m][n][a][suffix]["initial"] - latest]

                    if len(suffixes) > 1:
                        legend = "{} - {} - {} - {}".format(m, n, a,
                                                            suffix_abrs[
                                                                suffix]) \
                            if len(maps) > 1 \
                            else "{} - {} - {}".format(n,
                                                       a,
                                                       suffix_abrs[suffix])
                    else:
                        legend = "{} - {} - {}".format(m, n, a) \
                            if len(maps) > 1 else "{} - {}".format(n, a)

                    # TODO: deleting the trick bellow
                    legends += [legend.replace("15 - ", '').
                                    replace(" - sp", '').replace(" - ap", '')]
    print("toto")
    ids = np.arange(len(legends))

    ax = plt.gca()  # Get the current `matplotlib.axes.Axes` instance on
    # the current figure matching the given keyword args, or create one.

    # Setting the axes' label font size
    # ax.tick_params(labelsize=FONTSIZE)
    # TODO: set back 15 as font size
    plt.xticks(fontsize=FONTSIZE)  # , 15)
    plt.yticks(fontsize=FONTSIZE)
    plt.xticks(ids, legends)  # , rotation=70)
    # ax.set_xticks(x, legends, rotation=70)

    rects1 = plt.bar(ids, extras, align='center', alpha=0.5, bottom=latests,
                     color=["blue", "blue", "blue", "blue", "blue",
                            "red", "red", "red", "red", "red",
                            "green", "green", "green", "green", "green"])
    rects2 = plt.bar(ids, latests, align='center', alpha=0.5,
                     color=["#0756B2", "#0756B2", "#0756B2", "#0756B2",
                            "#0756B2",
                            "#F9ACB4", "#F9ACB4", "#F9ACB4", "#F9ACB4",
                            "#F9ACB4",
                            "#78F777", "#78F777", "#78F777", "#78F777",
                            "#78F777"])

    plt.ylabel("Cross-entropy", fontsize=FONTSIZE)

    # plt.title("Validation costs at the begining and the end of training for "
    #           "A and 15 agents", fontsize=FONTSIZE)

    plt.legend((rects1[0], rects2[0],
                rects1[6], rects2[6]),
               ("SGD-pre's initial cost",
                "SGD-pre's final cost",
                "Adagrad-pre's initial cost",
                "Adagrad-pre's final cost"),
               fontsize=LGDFONTSIZE)

    def autolabel(rects1, rects2):
        """
        Attach a text label above each bar displaying its height
        """
        for i, rect in enumerate(rects1):
            height = rect.get_height() + rects2[i].get_height()
            h_factor = 1. if rect.get_height() - rects2[i].get_height() > 1 \
                else 1.05 if rect.get_height() - rects2[i].get_height() > 0.2 \
                else 1.1
            ax.text(rect.get_x() + rect.get_width() / 2., h_factor * height,
                    format_value(height),
                    ha='center', va='bottom', fontsize=20)

        for rect in rects2:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1. * height,
                    format_value(height),
                    ha='center', va='bottom', fontsize=20)

    def format_value(value):
        return "%2.1f" % value

    autolabel(rects1, rects2)

    plt.show()


def plot_av_pop_end_costs(maps: list,
                          nagts: list,
                          archs: list,
                          bptts: list,
                          pathcosts: str,
                          modeltype: str,
                          modelvariant: str,
                          pre: bool = False,
                          datasrc: str = ''):
    """
    Plot end costs averaged over the population sizes for the map passed in
    argument

    :param maps:
    :type maps: list
    :param nagts:
    :type nagts: list
    :param archs:
    :type archs: list
    :param bptts:
    :type bptts: list
    :param modeltype:
    :type modeltype:
    :param modelvariant:
    :type modelvariant:
    :param pathcosts:
    :type pathcosts:
    :param pre:
    :type pre:
    :param datasrc: source whence data were drawn
    :type datasrc: str
    :return:
    :rtype:
    """

    if pre:
        nagts = ["pre"] + nagts

    # Dictionary of costs
    c_dic = misc.get_end_costs(maps=maps, nagts=nagts, archs=archs,
                               bptts=bptts, modeltype=modeltype,
                               modelvariant=modelvariant,
                               pathcosts=pathcosts, datasrc=datasrc)

    # Array of legends for each bar
    legends = []
    # Array of the top stacked bars' values: extra value with respect to the
    # latest one
    extras_av = []
    # Array of the bottom stacked bars' values: the latest costs
    latests_av = []

    for m in maps:
        for a in archs:
            # The current architecture's latest cost
            latest_av = np.average([c_dic[m][n][a]["latest"] for n in
                                    nagts])
            latests_av += [latest_av]
            # Difference between the initial costs and the latest
            # corresponding to the quantity needed to plot a stacked bar
            # chart
            extras_av += [np.average([c_dic[m][n][a]["initial"] for n in
                                      nagts]) - latest_av]
            legends += ["{} - {}".format(m, a) if len(maps) > 1
                        else "{}".format(a)]

    x = np.arange(len(legends))

    rects1 = plt.bar(x, extras_av, align='center', alpha=0.5,
                     bottom=latests_av, color="blue")
    rects2 = plt.bar(x, latests_av, align='center', alpha=0.5, color="red")

    ax = plt.gca()  # Get the current `matplotlib.axes.Axes` instance on
    # the current figure matching the given keyword args, or create one.

    # Setting the axes' label font size
    # ax.tick_params(labelsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    plt.xticks(x, legends, rotation=70)
    # ax.set_xticks(x, legends, rotation=70)

    plt.ylabel("Cost", fontsize=FONTSIZE)
    plt.title("Cost amount at the begining and the end of training for "
              "{}".format(', '.join(maps)), fontsize=TITLEFONTSIZE)
    plt.legend((rects1[0], rects2[0]), ('Initial cost', 'Final cost'),
               fontsize=LGDFONTSIZE)

    def autolabel(rects1, rects2):
        """
        Attach a text label above each bar displaying its height
        """
        for i, rect in enumerate(rects1):
            height = rect.get_height() + rects2[i].get_height()
            h_factor = 1. if rect.get_height() - rects2[i].get_height() > 1 \
                else 1.05 if rect.get_height() - rects2[i].get_height() > 0.2 \
                else 1.1
            ax.text(rect.get_x() + rect.get_width() / 2., h_factor * height,
                    format_value(height),
                    ha='center', va='bottom', fontsize=FONTSIZE)

        for rect in rects2:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1. * height,
                    format_value(height),
                    ha='center', va='bottom', fontsize=FONTSIZE)

    def format_value(value):
        return "%2.1f" % value

    autolabel(rects1, rects2)

    plt.show()


def plot_av_pop_maps_end_costs(maps: list,
                               nagts: list,
                               archs: list,
                               bptts: list,
                               pathcosts: str,
                               modeltype: str,
                               modelvariant: str,
                               pre: bool = False,
                               datasrc: str = ''):
    """
    Plot end costs averaged over the population sizes and the maps

    :param maps:
    :type maps: list
    :param nagts:
    :type nagts: list
    :param archs:
    :type archs: list
    :param bptts:
    :type bptts: list
    :param modeltype:
    :type modeltype:
    :param modelvariant:
    :type modelvariant:
    :param pathcosts:
    :type pathcosts:
    :param pre:
    :type pre:
    :param datasrc: source whence data were drawn
    :type datasrc: str
    :return:
    :rtype:
    """

    if pre:
        nagts = ["pre"] + nagts

    # Dictionary of costs
    c_dic = misc.get_end_costs(maps=maps, nagts=nagts, archs=archs,
                               bptts=bptts, modeltype=modeltype,
                               modelvariant=modelvariant,
                               pathcosts=pathcosts, datasrc=datasrc)

    # Array of legends for each bar
    legends = []
    # Array of the top stacked bars' values: extra value with respect to the
    # latest one
    extras_av = []
    # Array of the bottom stacked bars' values: the latest costs
    latests_av = []

    for a in archs:
        # The current architecture's averaged latest cost
        av_latest = np.average([c_dic[m][n][a]["latest"]
                                for m in maps for n in nagts])

        latests_av += [av_latest]
        # Averaged difference between the initial costs and the latest 
        # corresponding to the quantity needed to plot a stacked bar chart
        extras_av += [np.average([c_dic[m][n][a]["initial"]
                                  for m in maps for n in nagts]) - av_latest]

        legends += ["{}".format(a)]

    x = np.arange(len(legends))

    rects1 = plt.bar(x, extras_av, align='center', alpha=0.5,
                     bottom=latests_av,
                     color="blue")
    rects2 = plt.bar(x, latests_av, align='center', alpha=0.5, color="red")

    ax = plt.gca()  # Get the current `matplotlib.axes.Axes` instance on
    # the current figure matching the given keyword args, or create one.

    # Setting the axes' label font size
    # ax.tick_params(labelsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    plt.xticks(x, legends, rotation=70)
    # ax.set_xticks(x, legends, rotation=70)

    plt.ylabel("Cost", fontsize=FONTSIZE)
    plt.title("Cost amount at the begining and at the end of training for "
              "{} averaged over the population sizes and maps" \
              .format(', '''.join(maps)))
    plt.legend((rects1[0], rects2[0]), ('Initial cost', 'Final cost'),
               fontsize=LGDFONTSIZE)

    def autolabel(rects1, rects2):
        """
        Attach a text label above each bar displaying its height
        """
        for i, rect in enumerate(rects1):
            height = rect.get_height() + rects2[i].get_height()
            h_factor = 1. if rect.get_height() - rects2[i].get_height() > 1 \
                else 1.05 if rect.get_height() - rects2[i].get_height() > 0.2 \
                else 1.1
            ax.text(rect.get_x() + rect.get_width() / 2., h_factor * height,
                    format_value(height),
                    ha='center', va='bottom', fontsize=FONTSIZE, size=FONTSIZE)

        for rect in rects2:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1. * height,
                    format_value(height),
                    ha='center', va='bottom', fontsize=FONTSIZE, size=FONTSIZE)

    def format_value(value):
        return "%3.2f" % value

    autolabel(rects1, rects2)

    plt.show()


def plot_end_costs_suffixes_with_SGD_no_pre(maps: list,
                                            nagts: list,
                                            archs: list,
                                            bptts: list,
                                            pathcosts: str,
                                            modeltype: str,
                                            modelvariant: str,
                                            pre: bool = False,
                                            datasrc: str = ''):
    """

    :param maps:
    :type maps: list
    :param nagts:
    :type nagts: list
    :param archs:
    :type archs: list
    :param bptts:
    :type bptts: list
    :param modeltype:
    :type modeltype:
    :param modelvariant:
    :type modelvariant:
    :param pathcosts:
    :type pathcosts:
    :param pre:
    :type pre:
    :param datasrc: source whence data were drawn
    :type datasrc: str
    :rtype:
    """

    # Array of legends for each bar
    legends = []
    # Array of the top stacked bars' values: extra value with respect to the
    # latest one
    extras = []
    # Array of the bottom stacked bars' values: the latest costs
    latests = []

    # Suffixes
    suffixes = ["SGD-no_pre", "SGD-pre", "Adagrad-pre"]

    # Suffix abbreviations
    suffix_abrs = {"SGD-no_pre": "snp", "SGD-pre": "sp", "Adagrad-pre": "ap"}

    if pre:
        nagts = ["pre"] + nagts

    # Dictionary of costs
    c_dic = misc.get_end_costs(maps=maps, nagts=nagts, archs=archs,
                               bptts=bptts, modeltype=modeltype,
                               modelvariant=modelvariant,
                               pathcosts=pathcosts, datasrc=datasrc,
                               suffixes=suffixes)

    for suffix in suffixes:
        for m in maps:
            for n in nagts:
                for a in archs:
                    # The current architecture's latest cost
                    latest = c_dic[m][n][a][suffix]["latest"]
                    latests += [latest]
                    # Difference between the initial costs and the latest
                    # corresponding to the quantity needed to plot a stacked
                    # bar chart
                    extras += [c_dic[m][n][a][suffix]["initial"] - latest]

                    if len(suffixes) > 1:
                        legends += ["{} - {} - {} - {}".format(m, n, a,
                                                               suffix_abrs[
                                                                   suffix])
                                    if len(maps) > 1
                                    else "{} - {} - {}".format(n, a,
                                                               suffix_abrs[
                                                                   suffix])]
                    else:
                        legends += ["{} - {} - {}".format(m, n, a)
                                    if len(maps) > 1
                                    else "{} - {}".format(n, a)]
    ids = np.arange(len(legends))

    ax = plt.gca()  # Get the current `matplotlib.axes.Axes` instance on
    # the current figure matching the given keyword args, or create one.

    # Setting the axes' label font size
    # ax.tick_params(labelsize=FONTSIZE)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=FONTSIZE)
    plt.xticks(ids, legends, rotation=70)
    # ax.set_xticks(x, legends, rotation=70)

    rects1 = plt.bar(ids, extras, align='center', alpha=0.5, bottom=latests,
                     color=["blue", "blue", "blue", "blue", "blue",
                            "red", "red", "red", "red", "red",
                            "green", "green", "green", "green", "green"])
    rects2 = plt.bar(ids, latests, align='center', alpha=0.5,
                     color=["#0756B2", "#0756B2", "#0756B2", "#0756B2",
                            "#0756B2",
                            "#F9ACB4", "#F9ACB4", "#F9ACB4", "#F9ACB4",
                            "#F9ACB4",
                            "#78F777", "#78F777", "#78F777", "#78F777",
                            "#78F777"])

    plt.ylabel("Cross-entropy", fontsize=FONTSIZE)

    plt.title("Validation costs at the begining and the end of training for "
              "A and 15 agents", fontsize=FONTSIZE)

    plt.legend((rects1[0], rects2[0],
                rects1[6], rects2[6],
                rects1[11], rects2[1]),
               ("SGD-no-pre's initial cost",
                "SGD-no-pre's final cost",
                "SGD-pre's initial cost",
                "SGD-pre's final cost",
                "Adagrad-pre's initial cost",
                "Adagrad-pre's final cost"),
               fontsize=LGDFONTSIZE)

    def autolabel(rects1, rects2):
        """
        Attach a text label above each bar displaying its height
        """
        for i, rect in enumerate(rects1):
            height = rect.get_height() + rects2[i].get_height()
            h_factor = 1. if rect.get_height() - rects2[i].get_height() > 1 \
                else 1.05 if rect.get_height() - rects2[i].get_height() > 0.2 \
                else 1.1
            ax.text(rect.get_x() + rect.get_width() / 2., h_factor * height,
                    format_value(height),
                    ha='center', va='bottom', fontsize=FONTSIZE)

        for rect in rects2:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1. * height,
                    format_value(height),
                    ha='center', va='bottom', fontsize=FONTSIZE)

    def format_value(value):
        return "%2.1f" % value

    autolabel(rects1, rects2)

    plt.show()

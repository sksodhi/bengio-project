#!/usr/bin/python

#
# To start the program run
# ./main.py
# OR
# ./main.py -d
#
#

#    ____
#   /\  _`\                          __
#   \ \ \L\ \     __    ___      __ /\_\    ___
#    \ \  _ <'  /'__`\/' _ `\  /'_ `\/\ \  / __`\
#     \ \ \L\ \/\  __//\ \/\ \/\ \L\ \ \ \/\ \L\ \
#      \ \____/\ \____\ \_\ \_\ \____ \ \_\ \____/
#       \/___/  \/____/\/_/\/_/\/___L\ \/_/\/___/
#                               /\____/
#                                \_/__/
#     ______                                ____                                   __
#    /\__  _\                              /\  _`\               __               /\ \__
#    \/_/\ \/    __     __      ___ ___    \ \ \L\ \_ __   ___  /\_\     __    ___\ \ ,_\
#       \ \ \  /'__`\ /'__`\  /' __` __`\   \ \ ,__/\`'__\/ __`\\/\ \  /'__`\ /'___\ \ \/
#        \ \ \/\  __//\ \L\.\_/\ \/\ \/\ \   \ \ \/\ \ \//\ \L\ \\ \ \/\  __//\ \__/\ \ \_
#         \ \_\ \____\ \__/.\_\ \_\ \_\ \_\   \ \_\ \ \_\\ \____/_\ \ \ \____\ \____\\ \__\
#          \/_/\/____/\/__/\/_/\/_/\/_/\/_/    \/_/  \/_/ \/___//\ \_\ \/____/\/____/ \/__/
#                                                               \ \____/
#                                                                \/___/
#  http://patorjk.com/software/taag
#  Font Name: Larry 3D

import os
import os.path
import sys
import struct
import time
import cmd
import pickle
from array import array
import math
import numpy as np
import numpy.linalg as LA
import logging
import unittest
import getopt
import scipy.stats as ss
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import excel_utils
import read_dataset
import pca
import bayesian_classifier
import histogram_classifier
import linear_classifier
import confusion_matrix



FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
#logging.basicConfig(format=FORMAT,filename='./mldm.log',level=logging.DEBUG)
logging.basicConfig(format=FORMAT,stream=sys.stdout,level=logging.DEBUG)

scriptname = sys.argv[0]

def show_usage():
    print ("Usage:")
    print ("%s [options]" % scriptname)
    print ("options:")
    print ("  -h,--help                 : Show usage")
    print ("  -d,--debug                : Enable debugging")
    print ("  -p,--plot                 : plot graphs/pictures")
    print ("  -w,--writeback            : Write back results")

#
# Global variables
#
debug       = "no"
verbose     = "no"
plot        = "no"
writeback   = "no"


def get_options(argv):
    global debug
    global plot
    global verbose
    global writeback

    try:
       opts, args = getopt.getopt(argv,"dhpvw",["debug", "help", "plot", "verbose", "writeback"])
    except getopt.GetoptError:
       show_usage()
       sys.exit(2)
    for opt, arg in opts:
       if opt in ("-d", "--debug"):
          debug = "yes"
       elif opt in ("-p", "--plot"):
          plot = "yes"
       elif opt in ("-v", "--verbose"):
          verbose = "yes"
       elif opt in ("-w", "--writeback"):
          writeback = "yes"
       elif opt in ("-h", "--help"):
          show_usage()
          sys.exit()

    if debug == "yes":
        logging.debug('debug         : %s' % debug)
        logging.debug('plot          : %s' % plot)
        logging.debug('verbose       : %s' % verbose)
        logging.debug('writeback     : %s' % writeback)


def write_results(excelfile, sheet, \
                  num_pca, ppv_average, ppv_sum_all, ppv, \
                  min_ppv, min_ppv_class, \
                  max_ppv, max_ppv_class):
    print("Writing observation for %s classifier to excel sheet" % sheet)
    base = 1

    r = base + num_pca

    if sheet == 'Bayesian':
        ca_c = 2
    elif sheet == 'Histogram':
        ca_c = 3
    elif sheet == 'Linear':
        ca_c = 4
    else:
        logging.error("Invalid sheet: %s" % sheet)

    excel_utils.writeExcelData([num_pca], excelfile, sheet, r, 1)
    excel_utils.writeExcelData([num_pca], excelfile, 'CompareAllClassifiers', r, 1)
    print("Written num_pca")
    excel_utils.writeExcelData([ppv_average], excelfile, sheet, r, 2)
    excel_utils.writeExcelData([ppv_average], excelfile, 'CompareAllClassifiers', r, ca_c)
    print("Written ppv_average")
    excel_utils.writeExcelData([ppv_sum_all], excelfile, sheet, r, 3)
    print("Written ppv_sum_all")

    excel_utils.writeExcelData([min_ppv], excelfile, sheet, r, 4)
    print("Written min_ppv")
    excel_utils.writeExcelData([min_ppv_class], excelfile, sheet, r, 5)
    print("Written min_ppv_class")
    excel_utils.writeExcelData([max_ppv], excelfile, sheet, r, 6)
    print("Written max_ppv")
    excel_utils.writeExcelData([max_ppv_class], excelfile, sheet, r, 7)
    print("Written max_ppv_class")

    ppv_row = np.zeros((1,len(ppv)),dtype=np.float)
    for i in range(len(ppv)):
        ppv_row[0][i] = ppv[i]
    excel_utils.writeExcelData(ppv_row, excelfile, sheet, r, 8)
    print("Written ppv")


def write_cm_ppv_details(excelfile, sheet, pca_dims, cm):
    ppv = cm.get_ppv()
    print ppv

    print "____min/max PPV___"

    min_ppv = cm.get_min_ppv()
    min_ppv_class = cm.get_min_ppv_class()
    max_ppv = cm.get_max_ppv()
    max_ppv_class = cm.get_max_ppv_class()

    print("min_ppv: {0}({1}) max_ppv: {2}({3})".\
          format(cm.get_min_ppv(), \
          cm.get_min_ppv_class(), \
          cm.get_max_ppv(), \
          cm.get_max_ppv_class()))

    print "____average/sum PPV___"
    ppv_average = cm.get_ppv_average()
    ppv_sum_all = cm.get_ppv_sum_all()
    print("average ppv: {0} sum ppv: {1}".\
          format(cm.get_ppv_average(), \
          cm.get_ppv_sum_all()))

    write_results(excelfile, sheet, \
                  pca_dims, ppv_average, ppv_sum_all, ppv, \
                  min_ppv, min_ppv_class, \
                  max_ppv, max_ppv_class)

 


#
# Command line interface
#
class mldmCommandLine(cmd.Cmd):
    global debug

    """MLDM command processor. """
    intro = "Welcome to the MLDM shell. "
    intro += "Type help or ? to list commands.\n"

    def set_X_and_T(self, X, T):
        assert len(T) == len(X)
        self._X = X;
        self._T = T;
        self._num_classes = len(set(T))
        self._class_labels = set(T)

        #
        # default:
        #
        self._Xtraining = self._X  #
        self._Xtest = self._X
        self._Ttraining = self._T
        self._Ttest = self._T

        self._XtrainingPCA = self._Xtraining # No PCA by default
        self._XtestPCA = self._Xtraining # No PCA by default
        self._Ntraining = len(self._Xtraining)
        self._Ntest = len(self._Xtest)

    def do_split_data(self, line):
        """ Split data into training and test"""
        args = line.split(" ");
        percentage_str = args[0]

        if (len(args) == 1) and percentage_str:
            percentage = int(percentage_str)

            self._Xtraining, self._Xtest, self._Ttraining, self._Ttest = \
                linear_classifier.splitdata(self._X, self._T, percentage)

            self._Ntraining = len(self._Xtraining)
            self._Ntest = len(self._Xtest)

            """
            if writeback == "yes":
                excel_utils.writeExcelData([self._Ntraining], excelfile, 'TrainingData', 1, 2)
                print("Written Ntraining")
                excel_utils.writeExcelData(self._Ttraining, excelfile, 'TrainingData', 2, 1)
                print("Written Ttraining")
                excel_utils.writeExcelData(self._Xtraining, excelfile, 'TrainingData', 2, 2)
                print("Written Xtraining")
                excel_utils.writeExcelData([self._Ntest], excelfile, 'TestData', 1, 2)
                print("Written Ntest")
                excel_utils.writeExcelData(self._Ttest, excelfile, 'TestData', 2, 1)
                print("Written Ttest")
                excel_utils.writeExcelData(self._Xtest, excelfile, 'TestData', 2, 2)
                print("Written Xtest")
            """
        else:
            print("Usage: split_data <percentage>")

    def do_pca(self, line):
        """ Perform PCA"""

        args = line.split(" ");
        num_pc = args[0]

        if (len(args) == 1) and num_pc:
            self._pca_dims = int(num_pc)

            print("____ PCA : start ___________")

            print("Perfoming PCA (Principal componets: %d)" % self._pca_dims)
            self._XtrainingPCA = pca.perform_pca(self._Xtraining, self._pca_dims)
            self._XtestPCA     = pca.perform_pca(self._Xtest    , self._pca_dims)
            print("____ PCA : complete ________")

            if debug == "yes":
                print ("_XtrainingPCA.shape:",self._XtrainingPCA.shape)

        else:
            print("Usage: pca <num-principal-components>")

    def do_build_bc(self, line):
        """  Build Bayesian Classifier"""

        print("_______ Building Bayesian Classifier: start _______")

        try: 
            self._bc = bayesian_classifier.BayesianClassifier(self._XtrainingPCA, \
                                                              self._Ttraining, \
                                                              debug)
            self._bc.build()
        except Exception as e:
            logging.exception("message")
            loggin.debug("Not able to build Bayesian classifier ")
            sys.exc_clear()
        else:
            logging.debug("Bayesian classifier built")

        print("_______ Building Bayesian Classifier: done ________")

    def do_apply_bc(self, line):
        """ Apply Histogram Classifier"""
        print("_______ Applying Bayesian Classifier: start ______")
        self._bc_TresultsXtest  = self._bc.classify(self._XtestPCA)
        #print self._bc_TresultsXtest
        self._bc_cm = confusion_matrix.ConfusionMatrix(self._num_classes, \
                                                       self._class_labels, \
                                                       self._XtestPCA, \
                                                       self._Ttest, \
                                                       self._bc_TresultsXtest, \
                                                       debug)
        self._bc_cm.build()

        print("_______ Confusion Matrix __________")
        self._bc_cm.print_conf_matrix()
        if plot == "yes":
            self._bc_cm.plot_conf_matrix()
        print("_____________ PPV _________________")
        print self._bc_cm.get_ppv()
        if writeback == "yes":
            write_cm_ppv_details(excelfile, 'Bayesian', self._pca_dims, self._bc_cm)

        print "_______ Applying Bayesian Classifier: done _______"

    def do_build_hc(self, line):
        """ Build Histogram Classifier"""

        print "_______ Building Histogram Classifier: start ______"
        print "Good work needs time. Please be patient..."

        self._hc = histogram_classifier.HistogramClassifier(self._XtrainingPCA,\
                                                            self._Ttraining, \
                                                            debug)
        self._hc.build()
        print "_______ Building Histogram Classifier: done _______"

    def do_apply_hc(self, line):
        """ Apply Histogram Classifier"""
        print "_______ Applying Histogram Classifier: start ______"
        print "Good work needs time. Please be patient..."
        self._hc_TresultsXtest  = self._hc.classify(self._XtestPCA)
        #print self._hc_TresultsXtest
        self._hc_cm = confusion_matrix.ConfusionMatrix(self._num_classes,
                                                       self._class_labels,
                                                       self._XtestPCA,
                                                       self._Ttest,
                                                       self._hc_TresultsXtest,
                                                       debug)
        self._hc_cm.build()

        print "_______ Confusion Matrix __________"
        self._hc_cm.print_conf_matrix()
        if plot == "yes":
            self._hc_cm.plot_conf_matrix()
        print "_____________ PPV _________________"
        print self._hc_cm.get_ppv()
        if writeback == "yes":
            write_cm_ppv_details(excelfile, 'Histogram', self._pca_dims, self._hc_cm)

        print "_______ Applying Histogram Classifier: done _______"

    def do_build_lc(self, line):
        """ Build Linear Classifier"""

        print "_______ Building Linear Classifier: start _________"
        self._lc = linear_classifier.LinearClassifier(self._XtrainingPCA, \
                                                      self._Ttraining, debug)
        self._lc.build()
        print "_______ Building Linear Classifier: done __________"

    def do_apply_lc(self, line):
        """ Apply Linear Classifier"""

        print "_______ Applying Linear Classifier: start _________"
        self._lc_TresultsXtest = self._lc.classify(self._XtestPCA)

        #if debug == "yes":
        #    print self._lc_TresultsXtest
        self._lc_cm = confusion_matrix.ConfusionMatrix(self._num_classes, \
                                                       self._class_labels, \
                                                       self._XtestPCA, \
                                                       self._Ttest, \
                                                       self._lc_TresultsXtest, \
                                                       debug)
        self._lc_cm.build()
        print "_______ Confusion Matrix __________"
        self._lc_cm.print_conf_matrix()
        if plot == "yes":
            self._lc_cm.plot_conf_matrix()
        print "_____________ PPV _________________"

        if writeback == "yes":
            write_cm_ppv_details(excelfile, 'Linear', self._pca_dims, self._lc_cm)

        print "_______ Applying Linear Classifier: done __________"

    def do_build_apply_all(self, line):
        "Unit test"

        args = line.split(" ");
        num_pc = args[0]

        percentage_str = None
        if (len(args) == 2):
            percentage_str = args[1]

        if (len(args) >= 1) and num_pc:
            commands = []

            if percentage_str:
                percentage = int(percentage_str)
                if percentage < 1 or percentage > 99:
                    print ("Invalid percentage (%d). Will not re-split." % percentage)
                else:
                    commands.append("split_data " + percentage_str)

            commands.append("pca " + num_pc)

            commands.append("build_bc")
            commands.append("apply_bc")
            commands.append("build_hc")
            commands.append("apply_hc")
            commands.append("build_lc")
            commands.append("apply_lc")
            self.cmdqueue.extend(commands)
        else:
            print("Usage: build_apply_all <num-principal-components> " + \
                  "[<data-percentage-to-use-for-training>]")

    def do_ut_bc(self, line):
        "unit test linear classifier"

        commands = []
        commands.append("pca 16")
        commands.append("build_bc")
        commands.append("apply_bc")
        self.cmdqueue.extend(commands)

    def do_ut_hc(self, line):
        "unit test histogram classifier"

        commands = []
        commands.append("pca 16")
        commands.append("build_hc")
        commands.append("apply_hc")
        self.cmdqueue.extend(commands)

    def do_ut_lc(self, line):
        "unit test linear classifier"

        commands = []
        commands.append("pca 16")
        commands.append("build_lc")
        commands.append("apply_lc")
        self.cmdqueue.extend(commands)

    def do_ut(self, line):
        "Unit test"

        commands = []
        commands.append("build_apply_all 16 75")
        #commands.append("build_apply_all 16")
        commandline.cmdqueue.extend(commands)

    def do_collect_data(self, line):
        "Collect data"

        commands = []
        commands.append("split_data 75")
        for num_pc in range(1, 17):
            commands.append("pca " + str(num_pc))

            commands.append("build_bc")
            commands.append("apply_bc")
            commands.append("build_hc")
            commands.append("apply_hc")
            commands.append("build_lc")
            commands.append("apply_lc")

        commandline.cmdqueue.extend(commands)

    def do_quit(self, line):
        """ Quit """
        return True

#
# Main : Bengio Team Project
#
if __name__ == '__main__':
    get_options(sys.argv[1:])

    print("_____ Machine learning and data mining - Bengio Team projet ______")
 
    if writeback == "yes":
        excelfile=r"./Bengio_team_project_observations.xlsx";
        sheets=excel_utils.getSheetNames(excelfile)
        print "____ Sheets ____"
        print sheets

    print("Reading in Letter Recognition Dataset")
    X,T = read_dataset.read_letter_recognition_dataset( \
                     './datasets/letter-recognition/letter-recognition.data')
    num_T = len(T)
    set_T = set(T)
    num_classes = len(set_T)
    print "Read in " + str( num_T ) + " feature vectors, and " + \
                       str( num_classes ) + " classes"

    if debug == "yes":
        print ("len(T):", len(T))
        print ("set(T):",set(T))
        print ("num_classes:",num_classes)
        print ("X.shape:", X.shape)

    commandline = mldmCommandLine()
    commandline.set_X_and_T(X, T)
    commandline.prompt = "mldm >"
    commandline.cmdloop()

    sys.exit(0)

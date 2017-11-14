

import numpy as np
import matplotlib.pyplot as plt

#
# Confusion Matrix
#
class ConfusionMatrix():
    def __init__(self, num_class_labels, class_labels, Xtest, Ttest, Tresults, debug):
        assert len(Ttest) == len(Xtest)
        assert len(Ttest) == len(Tresults)
        assert num_class_labels == len(class_labels)

        assert num_class_labels >= len(set(Ttest))
        assert num_class_labels >= len(set(Tresults))

        self._num_class_labels = num_class_labels
        self._Xtest = Xtest
        self._Ttest = Ttest
        self._Tresults = Tresults
        self._debug = debug
        self._class_labels = np.sort(list(class_labels))  # labels vector

    #
    # Build Confusion Matrix
    #
    def build(self):
        print ("ConfusionMatrix: building confusion matrix")

        self._conf_mtrx = np.zeros((self._num_class_labels, self._num_class_labels)).astype('int32')

        if self._debug == "yes":
            print " _________ self._conf_mtrx: before ____________ "
            print self._conf_mtrx

        for i in range(len(self._Ttest)):
            row = list(self._class_labels).index(self._Ttest[i])
            col = list(self._class_labels).index(self._Tresults[i])

            """
            if self._debug == "yes":
                print ("row: ", row)
                print ("col: ", col)
                print ("self._Ttest[i]: ", self._Ttest[i])
                print ("self._Tresults[i]: ", self._Tresults[i])
            """
            self._conf_mtrx[row][col] +=1

        if self._debug == "yes":
            print " _________ self._conf_mtrx: after ____________ "
            print self._conf_mtrx

        self._ppv = np.zeros(self._num_class_labels).astype('float')

        for i in range(self._num_class_labels):
            numerator = self._conf_mtrx[i][i]

            total = 0
            for j in range(self._num_class_labels):
                total += self._conf_mtrx[j][i]

            """
            if self._debug == "yes":
                print ("class: ",i)
                print ("numerator: ",numerator)
                print ("total: ",total)
            """
            #
            # Hack for now
            if (total == 0) :
                total = 1

            self._ppv[i] = ((100.0 * numerator)/total)

        if self._debug == "yes":
            print " ____________ self._ppv _________ "
            print self._ppv

        self._min_ppv = self._ppv[0]
        self._min_ppv_class = self._class_labels[0]
        self._max_ppv = self._ppv[0]
        self._max_ppv_class = self._class_labels[0]

        for i in range(self._num_class_labels):
            if self._ppv[i] > self._max_ppv:
                self._max_ppv = self._ppv[i]
                self._max_ppv_class = self._class_labels[i]
            if self._ppv[i] < self._min_ppv:
                self._min_ppv = self._ppv[i]
                self._min_ppv_class = self._class_labels[i]

        if self._debug == "yes":
            print ("self._min_ppv:",self._min_ppv)
            print ("self._min_ppv_class:",self._min_ppv_class)
            print ("self._max_ppv:",self._max_ppv)
            print ("self._max_ppv_class:",self._max_ppv_class)

    def get_conf_matrix(self):
        print ("ConfusionMatrix: return ppv for class a particular class label")
        return self._conf_mtrx

    def print_conf_matrix(self):
        line = " "
        for e in (self._class_labels):
            line = line + "   " + e
        print line
        for g,e in enumerate(self._class_labels):
            line = e + " "
            for p in range(len(self._class_labels)):
                line = line + repr(self._conf_mtrx[ g, p ]).rjust(3) + " "
            print line

    def plot_conf_matrix(self):
        plt.matshow(self._conf_mtrx)
        plt.show()


    def get_ppv(self):
        print ("ConfusionMatrix: return ppv")
        return self._ppv

    def get_min_ppv(self):
        return self._min_ppv

    def get_min_ppv_class(self):
        return self._min_ppv_class

    def get_max_ppv(self):
        return self._max_ppv

    def get_max_ppv_class(self):
        return self._max_ppv_class

    def get_ppv_average(self):
        return np.average(self._ppv, axis=0)

    def get_ppv_sum_all(self):
        return np.sum(self._ppv, axis=0)

#!/usr/bin/python


def readExcelSheet1(excelfile):
    from pandas import read_excel
    return (read_excel(excelfile)).values


def readExcelRange(excelfile,sheetname="Sheet1",startrow=1,endrow=1,startcol=1,endcol=1):
    from pandas import read_excel
    values=(read_excel(excelfile, sheetname,header=None)).values;
    return values[startrow-1:endrow,startcol-1:endcol]



def readExcel(excelfile,**args):
    if args:
        data=readExcelRange(excelfile,**args)
    else:
        data=readExcelSheet1(excelfile)
    if data.shape==(1,1):
        return data[0,0]
    elif (data.shape)[0]==1:
        return data[0]
    else:
        return data


def writeExcelData(x,excelfile,sheetname,startrow,startcol):
    from pandas import DataFrame, ExcelWriter
    from openpyxl import load_workbook
    df=DataFrame(x)
    book = load_workbook(excelfile)
    writer = ExcelWriter(excelfile, engine='openpyxl') 
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df.to_excel(writer, sheet_name=sheetname,startrow=startrow-1, startcol=startcol-1, header=False, index=False)
    writer.save()
    writer.close()


def getSheetNames(excelfile):
    from pandas import ExcelFile
    return (ExcelFile(excelfile)).sheet_names


def getArrayForType(X, T, _type, numD):
    global debug

    TEMP1 = np.zeros((50000, numD))

    j = 0
    for i in range(len(X)):
        if T[i] == _type:
           TEMP1[j] = X[i]
           j += 1
    
    if debug == "yes":
        logging.debug(" Size for type %s : %d  _________ " % (_type,j))

    TEMP2 = TEMP1[:j]

    #print TEMP2
    return TEMP2

#    ______                                                                 __        ______    
#   /\  _  \                  __                                           /\ \__    /\  ___\   
#   \ \ \L\ \    ____    ____/\_\     __     ___     ___ ___      __    ___\ \ ,_\   \ \ \__/   
#    \ \  __ \  /',__\  /',__\/\ \  /'_ `\ /' _ `\ /' __` __`\  /'__`\/' _ `\ \ \/    \ \___``\ 
#     \ \ \/\ \/\__, `\/\__, `\ \ \/\ \L\ \/\ \/\ \/\ \/\ \/\ \/\  __//\ \/\ \ \ \_    \/\ \L\ \
#      \ \_\ \_\/\____/\/\____/\ \_\ \____ \ \_\ \_\ \_\ \_\ \_\ \____\ \_\ \_\ \__\    \ \____/
#       \/_/\/_/\/___/  \/___/  \/_/\/___L\ \/_/\/_/\/_/\/_/\/_/\/____/\/_/\/_/\/__/     \/___/ 
#				      /\____/                                                   
#                                     \_/__/                                                    
#
#  http://patorjk.com/software/taag 
#  Font Name: Larry 3D
#

#
# Main : Assigment 5 : Expectation Maximization 
#
if __name__ == '__main__':

    get_options(sys.argv[1:])

    print "_____ Machine learning and data mining - assingment5 ______"


    excelfile=r"/home/sandesh/machine_learning/mldm/assignment5/Assignment_5_Data_and_Template.xlsx";

    sheets=getSheetNames(excelfile);sheets

    print "____ Sheets ____"
    print sheets

    data=readExcel(excelfile)
    Xtrain=np.array(data[:,0:2],dtype=float);

    print "____ Xtrain.shape ____"
    print Xtrain.shape

    print "Displaying first 20"
    for i in range(20):
        print Xtrain[i]

    # fit a Gaussian Mixture Model with two components

    # generate random sample, two components
    np.random.seed(0)


    clf = mixture.GaussianMixture(n_components=3, covariance_type='full')
    """
    clf = mixture.GaussianMixture(n_components=3, 
                                  covariance_type='full',
                                  tol=0.00001,
                                  reg_covar=1e-06,
                                  max_iter=100000,
                                  n_init=1,
                                  init_params='kmeans',
                                  weights_init=None,
                                  means_init=None,
                                  precisions_init=None,
                                  random_state=None,
                                  warm_start=False,
                                  verbose=0,
                                  verbose_interval=10)
    """
    clf = clf.fit(Xtrain)


    Lpredict = clf.predict(Xtrain)
    print "____ Lpredict.shape ____"
    print Lpredict.shape
    print "Displaying first 20"
    for i in range(20):
        print Lpredict[i]

    label_counts = np.zeros((3))
    for i in range(len(Lpredict)):
        label_counts[Lpredict[i]] += 1

    print("label_counts:",label_counts)
    print np.sum(label_counts)

    PPpredict = clf.predict_proba(Xtrain)
    print "____ PPpredict.shape ____"
    print PPpredict.shape
    print "Displaying first 10"

    if debug == "yes":
        for i in range(10):
            print PPpredict[i][Lpredict[i]]*100

    if plot == "yes":

        print "Plotting..."
        #
        # Gets rid of the error
        # _tkinter.TclError: no display name and no $DISPLAY environment variable
        # 
        plt.switch_backend('agg')
        plt.close('all')
        fig = plt.figure()

        # display predicted scores by the model as a contour plot
        x = np.linspace(10., 80.)
        y = np.linspace(10., 80.)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = clf.score_samples(XX)

        print "____ Z.shape ____"
        print Z.shape

        Z = Z.reshape(X.shape)

        print "____ Z.shape ____"
        print Z.shape
        CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                         levels=np.logspace(0, 3, 10))
        CB = plt.colorbar(CS, shrink=0.5, extend='both')
        plt.scatter(Xtrain[:, 0], Xtrain[:, 1], .5)

        plt.title('Data clustering predicted by a GMM')
        plt.axis('tight')
        #plt.show()
        fig.savefig("./Xgmm_plot.png");

    ml = 2
    fl = 0
    cl = 1
    print ("Few males")
    for i in range(50):
        if Lpredict[i] == ml:
            print (Xtrain[i], Lpredict[i])

    print ("Few Females")
    for i in range(50):
        if Lpredict[i] == fl:
            print (Xtrain[i], Lpredict[i])

    print ("Few Children")
    for i in range(50):
        if Lpredict[i] == cl:
            print (Xtrain[i], Lpredict[i])



    if writeback == "yes":

        Confidence = np.zeros((len(Lpredict), 1),dtype=np.float)
        Lpredict_char_labels = np.chararray((len(Lpredict), 1))
        Lpredict_char_labels = np.chararray((len(Lpredict), 1))
 
        print "Writing back the results..."

        for i in range(len(Lpredict)):
            if Lpredict[i] == ml:
                CL = 'M'
            elif Lpredict[i] == fl:
                CL = 'F'
            elif Lpredict[i] == cl:
                CL = 'C'
            else:
                print "Error"
            Lpredict_char_labels[i][0] = CL
            Confidence[i][0] = PPpredict[i][Lpredict[i]]*100


        writeExcelData(Lpredict_char_labels, excelfile, 'Results', 2, 1)
        writeExcelData(Confidence, excelfile, 'Results', 2, 2)
        print("Written Class labels and predict probabilities")

        writeExcelData([label_counts[ml]], excelfile, 'Results', 2, 6) # Male
        writeExcelData([label_counts[fl]], excelfile, 'Results', 3, 6) # Female
        writeExcelData([label_counts[cl]], excelfile, 'Results', 4, 6) # Children
        print("Written class counts")


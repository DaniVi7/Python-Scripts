#!/usr/bin/python
############################################################################################################################################################
############################################################################################################################################################
import numpy as np
from itertools import islice
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d.axes3d import Axes3D
import os

import math
from math import isnan
import scipy as scy
import seaborn as sns

############################################################################################################################################################
############################################################################################################################################################

parser = argparse.ArgumentParser()

parser.add_argument("-SM", "--Smoothing", help="The number of point considered before and after each data point for smoothing", type=int)
parser.add_argument('-ALL', '--Do_everything', help="Do everything", type=bool)
parser.add_argument("-B", '--Bulk', help="Is the simulation to be analysed a 3D bulk simulation?", type=bool)
parser.add_argument("-fr", "--Frequency", help="The period with which data point are extracted from file (inverse of frequency)", type=float)
parser.add_argument("-SC_L", "--SC_L", help="Requires for plotting of results for various SC_L", type=bool)
parser.add_argument("-ABN", "--All_Blob_Numbers", help="Requires for plotting of results for various SC_L", type=bool)
parser.add_argument("-SF", "--Skip_Fraction", help="Requires for plotting of results for various SC_L", type=float)
parser.add_argument("-PNP", "--Patch_No_Patch", help="Requires for plotting of results for various SC_L", type=bool)
parser.add_argument("-TNR", "--Test_New_Rule", help="Requires for plotting of results for Testing of new Rule", type=bool) #!!DEBUGGING!!
parser.add_argument("-TSR", "--Test_SurfRe", help="Requires for plotting of results for Testing of Surface re-extraction", type=bool) #!!DEBUGGING!!


args = parser.parse_args()

############################################################################################################################################################

if args.Smoothing: #Parsing arguments
    SMOOTHING = args.Smoothing #string
else:
    SMOOTHING = 1000

if args.Do_everything: #Parsing arguments
    DOALL = True #string
else:
    DOALL = False

if args.SC_L: #Parsing arguments
    SCL = True #string
else:
    SCL = False

if args.All_Blob_Numbers: #Parsing arguments
    SCAN_BN = True #string
else:
    SCAN_BN = False

if args.Bulk: #Parsing arguments
    BULK = True #string
else:
    BULK = False

if args.Frequency: #Parsing arguments
    FREQUENCY = args.Frequency #string
else:
    FREQUENCY = 2

if args.Skip_Fraction: #Parsing arguments
    SKF = args.Skip_Fraction #string
else:
    SKF = 0.5

if args.Patch_No_Patch: #Parsing arguments
    PaNPa = True #string
else:
    PaNPa = False

if args.Test_New_Rule: #Parsing arguments   #!!DEBUGGING!!
    Te_NewRule = True #string   #!!DEBUGGING!!
else:   #!!DEBUGGING!!
    Te_NewRule = False  #!!DEBUGGING!!

if args.Test_SurfRe: #Parsing arguments   #!!DEBUGGING!!
    T_Surf_Re = True #string   #!!DEBUGGING!!
else:   #!!DEBUGGING!!
    T_Surf_Re = False  #!!DEBUGGING!!


############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################

def Read_evolution(FREQ):
    """Function reading standard evolution file from Sferocylinder simulations and extracting data. Returns arrays with data extracted
    with required Frequency
    Inputs: - FREQ, float: the inverse of the frequerncy with which data from evolution file should be sampled.
    - FNAME, string: the name of the evolution file from which data is to be extracted.
    Returns: - StepN_arr, float array: the array of the step number of all data points sampled;
             - En_arr, float array: the array of the energy of all data points sampled;
             - NBonds_arr, float array: the array of the number of bonds of all data points sampled;
             - NColl_arr, float array: the array of the number of adsorbed colloids of all data points sampled;
             - Theta_arr, float array: the array of Theta of all data points sampled.
    """
    with open("evolution.log") as EV:

        emptyline = EV.readline()

        #Initialising data arrays
        StepN_arr = []
        En_arr = []
        NColl_arr = []
        NAds_arr = []
        Theta_arr = []
        line_counter = 0

        while True:

            # print("line_counter = " + str(line_counter))
            try:
                thisline = EV.readline()
                # print(thisline)
                thisline = thisline.split()
                StepN = float(thisline[1][:-1])
                En = float(thisline[3][:-1])
                NColl = float(thisline[12][:-1])
                NAds = float(thisline[18][:-1])

                try:
                    Theta = float(thisline[21])
                except Exception:
                    Theta = float(thisline[21][:-1])

                else:
                    thisline = EV.readline() #Skipping faulty lines for now..
                    thisline = EV.readline()


                #I can add an else here to process Nrho for anomalous cases

            except Exception:
                #print("Reached end of file!")
                break

            else:
                if (line_counter%FREQ) == 0:
                    StepN_arr.append(StepN)
                    En_arr.append(En)
                    NColl_arr.append(NColl)
                    NAds_arr.append(NAds)
                    Theta_arr.append(Theta)


                line_counter += 1
                emptyline = EV.readline()
                emptyline = EV.readline()


        #"End of while loop"
        #print("Created arrays of data with required frequency!\n")



    return StepN_arr, En_arr, NColl_arr, NAds_arr, Theta_arr

##############################################################################

def Read_step_vs_AvgVers(FREQ):

    with open("step_vs_AvgVers.log") as AV:

        emptyline = AV.readline()
        emptyline = AV.readline()

        #Initialising data arrays
        StepVers_arr = []
        AvgVers_arr = []
        line_counter = 0

        while True:

            # print("line_counter = " + str(line_counter))
            try:
                thisline = AV.readline()
                # print(thisline)
                thisline = thisline.split()
                StepN = float(thisline[1])
                x = float(thisline[4])
                y = float(thisline[5])
                z = float(thisline[6])

                if isnan(x): #If no colloid was present at this step, skip it
                    continue

            except Exception:
                #print("Reached end of file!")
                break

            else:
                if (line_counter%FREQ) == 0:

                    vers = np.array([x, y, z])
                    AvgVers_arr.append(vers)
                    StepVers_arr.append(StepN)

                line_counter += 1


        #"End of while loop"
        #print("Created arrays of data with required frequency!\n")


    return StepVers_arr, AvgVers_arr

##############################################################################

def Plot_data_smooth(STEPN_X, EN_Y, NCOLL_Y, THETA_Y, SM_RANGE):
    """Function reading standard evolution file from Sferocylinder simulations and extracting data. Returns arrays with data extracted
    with required Frequency
    Inputs: - STEPN_X, float array: the array of the step number of all data points sampled;
            - EN_Y, float array: the array of the energy of all data points sampled;
            - NBONDS_Y, float array: the array of the number of bonds of all data points sampled;
            - NCOLL_Y, float array: the array of the number of adsorbed colloids of all data points sampled;
            - PACK_Y, float array: the array of Packing Fraction of all data points sampled.
    Returns: None.
    """
    SM_EN_Y = []
    SM_NCOLL_Y = []
    SM_THETA_Y = []

    for i in range(SM_RANGE, (len(STEPN_X) - SM_RANGE)):
        SM_EN_Y.append(np.average(EN_Y[i - SM_RANGE : i + SM_RANGE] ) )
        SM_NCOLL_Y.append(np.average(NCOLL_Y[i - SM_RANGE : i + SM_RANGE] ) )
        SM_THETA_Y.append(np.average(THETA_Y[i - SM_RANGE : i + SM_RANGE] ) )

    last_half = int(len(STEPN_X)/2)
    ############################################################################
    AVG_En = np.average(EN_Y)
    toAVG_NCOLL = NCOLL_Y[last_half:]
    AVG_NColl = np.average(toAVG_NCOLL)
    toAVG_THETA = THETA_Y[last_half:]
    AVG_THETA = np.average(toAVG_THETA)

    #print("Plotting the Energy evolution.. ")
    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.set_title("Total Energy")
    ax1.set_ylabel("Energy")
    ax1.set_xlabel("Number of steps")
    ax1.plot(STEPN_X, EN_Y, color='r')
    ax1.axhline(y = AVG_En, color = 'r', linestyle = '-')

    ax2 = fig.add_subplot(212)
    ax2.set_title("Local Average of Total Energy")
    ax2.set_ylabel("Energy")
    ax2.set_xlabel("Number of steps")
    ax2.plot(STEPN_X[SM_RANGE:(len(STEPN_X) - SM_RANGE)], SM_EN_Y, color='r')
    ax2.axhline(y = AVG_En, color = 'r', linestyle = '-')

    fig.tight_layout()
    fig.savefig(("TotE_vs_Nstep.pdf"))
    plt.close()
    ############################################################################

    ############################################################################
    # AVG_NColl = np.average(NCOLL_Y)
    #print("Plotting the Number of Colloids evolution.. ")
    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.set_title("Number of Ads. Colloids (NC)")
    ax1.set_ylabel("NC")
    ax1.set_xlabel("Number of steps")
    ax1.plot(STEPN_X, NCOLL_Y, color='g')
    ax1.axhline(y = AVG_NColl, color = 'g', linestyle = '-')

    ax2 = fig.add_subplot(212)
    ax2.set_title("NC Local Average")
    ax2.set_ylabel("NC")
    ax2.set_xlabel("Number of steps")
    ax2.plot(STEPN_X[SM_RANGE:(len(STEPN_X) - SM_RANGE)], SM_NCOLL_Y, color='g')
    ax2.axhline(y = AVG_NColl, color = 'g', linestyle = '-')

    fig.tight_layout()
    fig.savefig(("NColl_vs_Nstep.pdf"))
    plt.close()
    ############################################################################

    ############################################################################
    # AVG_THETA = np.average(THETA_Y)
    #print("Plotting the Theta evolution.. ")
    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.set_title("Theta")
    ax1.set_ylabel("Theta")
    ax1.set_xlabel("Number of steps")
    ax1.plot(STEPN_X, THETA_Y, color='k')
    ax1.axhline(y = AVG_THETA, color = 'k', linestyle = '-')

    ax2 = fig.add_subplot(212)
    ax2.set_title("Theta Local Average")
    ax2.set_ylabel("Theta")
    ax2.set_xlabel("Number of steps")
    ax2.plot(STEPN_X[SM_RANGE:(len(STEPN_X) - SM_RANGE)], SM_THETA_Y, color='k')
    ax2.axhline(y = AVG_THETA, color = 'k', linestyle = '-')

    fig.tight_layout()
    plt.savefig(("Theta_vs_Nstep.pdf"))
    plt.close()
    ############################################################################

    with open("Averages.data", "w+") as A:
        A.write("Average Energy = " + str(AVG_En))
        A.write("\nAverage Number of Adsorbed colloids = " + str(AVG_NColl))
        A.write("\nAverage Theta = " + str(AVG_THETA) + "\n")
        A.write("Number of steps analysed = " + str(STEPN_X[-1]))


    return

##############################################################################

def Plot_Avg_Vers(STEPN_X, AVG_VERS_Y):
    """Function reading standard evolution file from Sferocylinder simulations and extracting data. Returns arrays with data extracted
    with required Frequency
    Inputs: - STEPN_X, float array: the array of the step number of all data points sampled;
            - EN_Y, float array: the array of the energy of all data points sampled;
            - NBONDS_Y, float array: the array of the number of bonds of all data points sampled;
            - NCOLL_Y, float array: the array of the number of adsorbed colloids of all data points sampled;
            - PACK_Y, float array: the array of Packing Fraction of all data points sampled.
    Returns: None.
    """
    last_half = int(len(STEPN_X)/2)
    AVG_VERS_Y = np.array(AVG_VERS_Y)
    #print("AVG_VERS_Y : ")
    #print(AVG_VERS_Y)
    ############################################################################
    toAVG_VERS = AVG_VERS_Y[last_half:, 2]
    #print("toAVG_VERS : ")
    #print(toAVG_VERS)

    AVG_AVG_Z = np.average(toAVG_VERS)
    #print("AVG_AVG_Z = " + str(AVG_AVG_Z))
    # AVG_THETA = np.average(THETA_Y)
    #print("Plotting the Theta evolution.. ")
    fig = plt.figure()

    ax1 = fig.add_subplot()
    ax1.set_title("Average Versor of SCs")
    ax1.set_ylabel("Versor Components")
    ax1.set_xlabel("Number of steps")
    ax1.plot(STEPN_X, AVG_VERS_Y[:, 0], color='b', label="X")
    ax1.plot(STEPN_X, AVG_VERS_Y[:, 1], color='r', label="y")
    ax1.plot(STEPN_X, AVG_VERS_Y[:, 2], color='g', label="Z")
    ax1.axhline(y = AVG_AVG_Z, color = 'k', linestyle = '-', label="Aerage Z")
    ax1.legend(loc="best")

    fig.tight_layout()
    plt.savefig(("AVG_vers_vs_Nstep.pdf"))
    plt.close()
    ############################################################################

    with open("Averages.data", "a+") as A:
        A.write("\nAverage Z of Versors = " + str(AVG_AVG_Z))

    return

##############################################################################

def Read_reports():
    """!!!!!! ONLY READING THETA FOR NOW !!!!!
    """

    with open("Averages.data") as AV:

        Energy_line = AV.readline()
        NColl_line = AV.readline()

        Theta_line = AV.readline().split()
        THETA = Theta_line[3]

        ignore_line = AV.readline()

        AVG_Z_line = AV.readline().split()
        AVG_Z = AVG_Z_line[5]

        return THETA, AVG_Z

##############################################################################

def Plot_for_all_nR(TITLE):

    name_of_dirs = os.listdir()
    nR_Theta_list = []
    AVGZ_list = []
    Restarted = False
    #print(name_of_dirs)

    for i in name_of_dirs:
        this_point = []
        try:
            aaa = float(i[3:])
        except Exception:
            #print("Reached end of file!")
            continue


        this_point.append(float(i[3:])) #ZZ value
        os.chdir(i) #entering new directory with data
        print("entering new directory: " + str(i))
        inside_dir = os.listdir()
        if "RE" in inside_dir:
            os.chdir("RE") #entering new directory with data
            Restarted = True
        ##

        try:
            AllSteps, AllEn, AllNColl, AllNAds, AllTheta = Read_evolution(FREQUENCY)
            VStep, AllAvgVers = Read_step_vs_AvgVers(FREQUENCY)
        except Exception:
            #print("Reached end of file!")
            os.chdir("../") #returning to initial directory
            print("exiting directory.. A")
            if Restarted==True:
                os.chdir("../") #returning to initial directory
                print("exiting directory.. B")

            continue
        else:
            if (len(AllSteps) > 1):
                #print("Check - 1")
                Plot_data_smooth(AllSteps, AllEn, AllNColl, AllTheta, SMOOTHING)
                #print("Check - 2")
                Plot_Avg_Vers(VStep, AllAvgVers)
                #print("Check - 3")

            else:
                print("!!!!! len(AllSteps) <= 1 !!!!!")
                print("AllSteps: ")
                print(AllSteps)
                print("\n")

            #print("Check - 4")
            this_theta, this_AvgZ = Read_reports()
            this_point.append(float(this_theta))
            #print("Check - 5")
            this_point = np.array(this_point) #Pack value

            nR_Theta_list.append(this_point)
            #print("this_AvgZ = " + str(this_AvgZ))
            AVGZ_list.append(float(this_AvgZ))
            #print("Check - 6")
            os.chdir("../") #returning to initial directory
            print("exiting directory.. C")

            if Restarted==True:
                os.chdir("../") #returning to initial directory
                print("exiting directory.. D")

        ##
    ##End of for loop over directories


    nR_Theta_list = np.array(nR_Theta_list)
    AVGZ_list = np.array(AVGZ_list)

    if (len(nR_Theta_list) == 0):
        raise ValueError

    ind = np.argsort(nR_Theta_list[:,0])
    nR_Theta_list = nR_Theta_list[ind]
    AVGZ_list = AVGZ_list[ind]

    #print("nR_Theta_list: ")
    #print(nR_Theta_list)


    with open("nR_vs_Theta_recap.data", "w+") as TNR:
        #for counti, thetaj in enumerate(Theta_list):
        #    TNR.write("nR = " + str(Density_list[counti]) + ", Theta = "+ str(thetaj) + "\n")

        TNR.write("nR    Theta \n")

        for counti, thetaj in enumerate(nR_Theta_list):
            #TNR.write(str(Dens_Theta_list[counti]) + "    "+ str(thetaj) + "\n")
            TNR.write(str(nR_Theta_list[counti,0]) + "    "+ str(nR_Theta_list[counti,1]) + "\n")

        TNR.close()

    plt.figure()
    plt.title(TITLE)
    plt.ylabel("Theta")
    plt.xlabel("nR")
    plt.plot(nR_Theta_list[:,0], nR_Theta_list[:,1], color='r', linestyle="-", marker="o")

    #plt.legend(loc="best")
    plt.savefig(("nR_vs_Theta.pdf"))
    plt.show()
    plt.close()

    return nR_Theta_list, AVGZ_list

##############################################################################

def Plot_for_all_SC_L(PTCH, BOND_EN, BLOB_N):

    dir_names = os.listdir()

    nR_vs_THETA_arr = []
    nR_vs_AVGZ = []
    title_string = PTCH + ", BF = -" + BOND_EN[-1] + ", NLig = " + BLOB_N[-1]
    file_string = PTCH + "_BF_" + BOND_EN[-1] + "_NLig_" + BLOB_N[-1]

    #label_list = ["SC_L_6", "SC_L_3", "SC_L_0", "SC_L_9", "SC_L_12", "SC_L_15"]
    label_list = ["SC_L_0", "SC_L_3", "SC_L_6", "SC_L_9", "SC_L_12", "SC_L_15"]
    # label_list = ["SC_L_6", "SC_L_3", "SC_L_0"]
    new_label_list = []
    for j in label_list:
    # for j in dir_names:
        #if (dir_names[:4] == "SC_L"):
        if j not in dir_names:
            continue

        title_string_2 = title_string + ", SC_L = " + j[-1]

        os.chdir(j)
        print("entering new directory: " + str(j))

        try:
            to_app, tapp_AVGZ = Plot_for_all_nR(title_string_2)
        except Exception:
            #print("Reached end of file!")
            os.chdir("../")
            print("exiting directory.. E")

            continue

        else:
            new_label_list.append(j)
            nR_vs_THETA_arr.append(np.array(to_app))
            nR_vs_AVGZ.append(tapp_AVGZ)
            os.chdir("../")
            print("exiting directory.. F")

        continue
    #End of SC_L loop

    nR_vs_THETA_arr = np.array(nR_vs_THETA_arr)
    nR_vs_AVGZ = np.array(nR_vs_AVGZ)

    plt.figure()
    plt.title(title_string)
    plt.ylabel("Theta")
    plt.xlabel("nR")
    for k in range(len(new_label_list)):

        plt.plot(nR_vs_THETA_arr[k][:, 0], nR_vs_THETA_arr[k][:, 1], linestyle="-", marker="o", label=new_label_list[k])

    plt.legend(loc="best")
    PDF_NAME_1 = "All_SC_L_Theta_vs_nR_" + file_string + ".pdf"
    #plt.savefig(("All_SC_L_Theta_vs_nR.pdf"))
    plt.savefig((PDF_NAME_1))

    plt.yscale("log")
    plt.xscale("log")
    PDF_NAME_2 = "All_SC_L_Theta_vs_nR_Log_" + file_string + ".pdf"
    #plt.savefig(("All_SC_L_Theta_vs_nR_Log.pdf"))
    plt.savefig((PDF_NAME_2))


    #plt.show()
    plt.close()

    ##################################################################################################

    plt.figure()
    plt.title(title_string)
    plt.ylabel("AVG Z")
    plt.xlabel("nR")
    for k in range(len(new_label_list)):
        #print("nR_vs_THETA_arr: ")
        #print(nR_vs_THETA_arr)

        #print("nR_vs_AVGZ: ")
        #print(nR_vs_AVGZ)

        #plt.plot(nR_vs_THETA_arr[k][:, 0], nR_vs_AVGZ[k, :], linestyle="-", marker="o", label=new_label_list[k])
        plt.plot(nR_vs_THETA_arr[k][:, 0], nR_vs_AVGZ[k], linestyle="-", marker="o", label=new_label_list[k])

    plt.legend(loc="best")
    PDF_NAME_2 = "All_SC_L_AVGZ_vs_nR_" + file_string + ".pdf"
    #plt.savefig(("All_SC_L_Theta_vs_nR.pdf"))
    plt.savefig((PDF_NAME_2))
    plt.close()


    return new_label_list, nR_vs_THETA_arr

###########################################################################################################################################################

def Plot_for_all_Rho(Wpatch, CURR_BN, WBF):

    dir_names = os.listdir()
    list_of_results = []
    Rho_label = ["rho_1e4", "rho_1e5", "rho_1e6"]
    for j in Rho_label:
    # for j in dir_names:
        #if (dir_names[:4] == "SC_L"):
        if j not in dir_names:
            continue

        os.chdir(j)
        print("entering new directory: " + str(j))

        SCL_Labels, SCL_nR_vs_Theta  = Plot_for_all_SC_L(Wpatch, WBF, CURR_BN)
        #Each element of the new output list will contain: at pos 0 the number of blobs per coll; at pos 1 the value of SCL of the sample; at pos 2 the array of nR vs Theta results for the specified parameters
        list_of_results.append(np.array([j, SCL_Labels, SCL_nR_vs_Theta]))

        os.chdir("../")
        print("exiting directory..")

        continue


    list_of_results = np.array(list_of_results)

    #Plot at same SC_L but different Blob N
    #print("list_of_results : " )
    #print(list_of_results)


    for countt, aspect in enumerate(SCL_Labels):
        plt.figure()
        curr_title = aspect + ", BF = " + str(WBF)
        plt.title(curr_title)
        plt.ylabel("Theta")
        plt.xlabel("nR")

        for k in range(len(list_of_results)):
            #if (len ())
            #print("list_of_results[k] : " )
            #print(list_of_results[k])

            #print("list_of_results[k][2] : " )
            #print(list_of_results[k][2])

            #print("list_of_results[k][2][countt] : " )
            #print(list_of_results[k][2][countt])

            try:
                chh = list_of_results[k][2][countt]
            except Exception:
                continue

            plt.plot(list_of_results[k][2][countt][:, 0], list_of_results[k][2][countt][:, 1], linestyle="-", marker="o", label=list_of_results[k][0])

        plt.legend(loc="best")
        PDF_NAME_1 = "All_Rho_" + CURR_BN + "_" + aspect +  "_Theta_vs_nR_" + curr_title + ".pdf"
        #plt.savefig(("All_SC_L_Theta_vs_nR.pdf"))
        plt.savefig((PDF_NAME_1))



    return list_of_results

###########################################################################################################################################################

def Plot_for_all_BN(Wpatch, WBF):

    dir_names = os.listdir()
    list_of_results = []
    # blobN_label = ["Blob_1", "Blob_3", "Blob_5"]
    blobN_label = ["Blob_5"]

    for bn in blobN_label:
    # for j in dir_names:
        #if (dir_names[:4] == "SC_L"):

        os.chdir(bn)
        print("entering new directory: " + str(bn))

        RES  = Plot_for_all_Rho(Wpatch, bn, WBF)
        #Each element of the new output list will contain: at pos 0 the number of blobs per coll; at pos 1 the value of SCL of the sample; at pos 2 the array of nR vs Theta results for the specified parameters
        # list_of_results.append(np.array([rho, SCL_Labels, SCL_nR_vs_Theta]))
        list_of_results.append(np.array([bn, RES]))

        os.chdir("../")
        print("exiting directory..")

        continue


    list_of_results = np.array(list_of_results)

    #Plot at same SC_L but different Blob N
    #print("list_of_results : " )
    #print(list_of_results)


    for countt, aspect in enumerate(list_of_results[0][1][0][1]): #SCL_LABELS
        plt.figure()
        curr_title = aspect + ", BF = " + str(WBF)
        plt.title(curr_title)
        plt.ylabel("Theta")
        plt.xlabel("nR")

        for k in range(len(blobN_label)):
            for rr in range(len(list_of_results[k][1])): # rho labels
                try:
                    this_rho = list_of_results[k][1][rr][0]
                    chh = list_of_results[k][1][rr][2][countt]
                    lbl = blobN_label[k] + "_" + this_rho
                except Exception:
                    continue

                plt.plot(list_of_results[k][1][rr][2][countt][:, 0], list_of_results[k][1][rr][2][countt][:, 1], linestyle="-", marker="o", label=lbl)

        plt.legend(loc="best")
        PDF_NAME_1 = "All_BN_" + aspect +  "_Theta_vs_nR_" + curr_title + ".pdf"
        #plt.savefig(("All_SC_L_Theta_vs_nR.pdf"))
        plt.savefig((PDF_NAME_1))



    return list_of_results

##############################################################################

def Plot_for_all_BF(Have_Patch):

    dir_names = os.listdir()
    BF_x_results = []

    for j in ["BF_4", "BF_3", "BF_2"]:
    # for j in ["BF_3"]:
    # for j in dir_names:
        #if (dir_names[:4] == "SC_L"):
        if j not in dir_names:
            continue

        os.chdir(j)
        print("entering new directory: " + str(j))
        try:
            ALLBn_x_result = Plot_for_all_BN(Have_Patch, j)
            BF_x_results.append(np.array([j, ALLBn_x_result]))
        except:
            os.chdir("../")
            print("exiting directory.. I")
            continue

        os.chdir("../")
        print("exiting directory.. G")

        continue

    BF_x_results = np.array(BF_x_results)

    #Write all results in list_of_results to file (this will be all results for a given ligand configuration (Patch / No Patch ))

    #Plot couples of SC_L, at fixed BN but different BF


    return BF_x_results

##############################################################################

def Plot_Patch_and_No_Patch():

    # dir_names = os.listdir()
    FINAL_RESULT_LIST = []

    #for j in ["Patch", "No_patch"]:
    for j in ["No_patch", "Patch"]:
    # for j in dir_names:
        #if (dir_names[:4] == "SC_L"):

        os.chdir(j)
        print("entering new directory: " + str(j))
        Configuration_results = Plot_for_all_BF(j)
        FINAL_RESULT_LIST.append(np.array([j, Configuration_results]))
        os.chdir("../")
        print("exiting directory.. H")

        continue

    FINAL_RESULT_LIST = np.array(FINAL_RESULT_LIST)

    #Plot comparison of Patch and No patch on same graph.. various combo

    return

##############################################################################
##############################################################################
##############################################################################

def Plot_for_New_Rule_Test():

    nR_vs_THETA_arr = []
    label_array = ["No_patch", "Patch"]

    for j in label_array:
    # for j in dir_names:
        #if (dir_names[:4] == "SC_L"):

        os.chdir(j)
        print("entering new directory: " + str(j))
        to_app, tapp_AVGZ = Plot_for_all_nR(label_array)

        nR_vs_THETA_arr.append(np.array(to_app))

        os.chdir("../")

    nR_vs_THETA_arr = np.array(nR_vs_THETA_arr)


    plt.figure()
    plt.title("Patch vs No Patch with 1 Blob x Colloid")
    plt.ylabel("Theta")
    plt.xlabel("nR")
    for k in range(len(label_array)):

        plt.plot(nR_vs_THETA_arr[k][:, 0], nR_vs_THETA_arr[k][:, 1], linestyle="-", marker="o", label=label_array[k])

    plt.legend(loc="best")
    PDF_NAME_1 = "Patch_vs_No_Patch_Theta_vs_nR_New_Rule.pdf"
    plt.savefig((PDF_NAME_1))
    plt.close()


    return

##############################################################################

def Plot_for_Surf_ext_Test():

    nR_vs_THETA_arr = []
    label_array = ["WITH", "WITHOUT"]

    for j in label_array:
    # for j in dir_names:
        #if (dir_names[:4] == "SC_L"):

        os.chdir(j)
        print("entering new directory: " + str(j))
        to_app, tapp_AVGZ = Plot_for_all_nR(label_array)

        nR_vs_THETA_arr.append(np.array(to_app))

        os.chdir("../")

    nR_vs_THETA_arr = np.array(nR_vs_THETA_arr)


    plt.figure()
    plt.title("With surface re-extraction vs Without with 1 Blob x Colloid")
    plt.ylabel("Theta")
    plt.xlabel("nR")
    for k in range(len(label_array)):

        plt.plot(nR_vs_THETA_arr[k][:, 0], nR_vs_THETA_arr[k][:, 1], linestyle="-", marker="o", label=label_array[k])

    plt.legend(loc="best")
    PDF_NAME_1 = "With_vs_Withou_Surf_re.pdf"
    plt.savefig((PDF_NAME_1))
    plt.close()


    return

##############################################################################
##############################################################################
##############################################################################

## MAIN ##

#AllSteps, AllEn, AllNColl, AllPack = Read_evolution3D(FREQUENCY)
#Plot_data_smooth3D(AllSteps, AllEn, AllNColl, AllPack, SMOOTHING)
if (SCL == True):
    Plot_for_all_SC_L()

elif (SCAN_BN):
    Plot_for_all_BN()
elif (PaNPa):
    Plot_Patch_and_No_Patch()

elif (Te_NewRule):    #!!DEBUGGING!!
    Plot_for_New_Rule_Test()    #!!DEBUGGING!!

elif (T_Surf_Re):    #!!DEBUGGING!!
    Plot_for_Surf_ext_Test()    #!!DEBUGGING!!

else:
    Plot_for_all_ZZ()

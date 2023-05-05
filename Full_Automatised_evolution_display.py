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
parser.add_argument("-PO", "--Plot_only", help="Requires for plotting of results for various SC_L", type=bool)
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
    FREQUENCY = 1

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

if args.Plot_only: #Parsing arguments
    PlotOnly = True #string
else:
    PlotOnly = False


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
#!
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
    last_third = int(len(STEPN_X)*2/3)
    ############################################################################
    AVG_En = np.average(EN_Y)

    if (len(STEPN_X) > 2e8):
        toAVG_NCOLL = NCOLL_Y[last_third:]
        toAVG_THETA = THETA_Y[last_third:]

    else:
        toAVG_NCOLL = NCOLL_Y[last_half:]
        toAVG_THETA = THETA_Y[last_half:]

    AVG_NColl = np.average(toAVG_NCOLL)
    AVG_THETA = np.average(toAVG_THETA)

    # AVG_NColl = np.average(NCOLL_Y)
    # AVG_THETA = np.average(THETA_Y)

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
#!
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
#!
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
                os.chdir("../") #returning to initial directory
                print("exiting directory.. J")
                continue


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

    plt.xscale("log")
    #plt.legend(loc="best")
    plt.savefig(("nR_vs_Theta.pdf"))
    plt.show()
    plt.close()

    return nR_Theta_list, AVGZ_list

##############################################################################

def Plot_for_all_SC_L(PTCH, BOND_EN, BLOB_N, NRHO):

    dir_names = os.listdir()

    nR_vs_THETA_arr = []
    nR_vs_AVGZ = []
    title_string = PTCH + ", BF = -" + BOND_EN[-1] + ", NLig = " + BLOB_N[-1] + ", NRho = 1e-" + NRHO[-1]
    file_string = PTCH + "_BF_" + BOND_EN[-1] + "_NLig_" + BLOB_N[-1] + "_NRho_1e" + NRHO[-1]

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

    plt.xscale("log")
    plt.legend(loc="best")
    PDF_NAME_1 = "All_SC_L_Theta_vs_nR_" + file_string + ".pdf"
    #plt.savefig(("All_SC_L_Theta_vs_nR.pdf"))
    plt.savefig((PDF_NAME_1))

    plt.yscale("log")
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

    plt.xscale("log")
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

        SCL_Labels, SCL_nR_vs_Theta  = Plot_for_all_SC_L(Wpatch, WBF, CURR_BN, j)
        #Each element of the new output list will contain: at pos 0 the number of blobs per coll; at pos 1 the value of SCL of the sample; at pos 2 the array of nR vs Theta results for the specified parameters
        list_of_results.append(np.array([j, SCL_Labels, SCL_nR_vs_Theta]))

        os.chdir("../")
        print("exiting directory..")

        continue


    list_of_results = np.array(list_of_results)

    #Plot at same SC_L but different Blob N
    #print("list_of_results : " )
    #print(list_of_results)

    # SCL_label_list = ["SC_L_0", "SC_L_3", "SC_L_6", "SC_L_9", "SC_L_12", "SC_L_15"]
    SCL_label_list = ["SC_L_0", "SC_L_3", "SC_L_6"]


    for countt, aspect in enumerate(SCL_label_list): #SCL_Labels
        plt.figure()
        curr_title = aspect + ", BF = " + str(WBF)
        plt.title(curr_title)
        plt.ylabel("Theta")
        plt.xlabel("nR")
        broken = 0

        for k in range(len(list_of_results)):
            try:
                chh = list_of_results[k][2][countt]
            except Exception:
                broken += 1
                continue

            plt.plot(list_of_results[k][2][countt][:, 0], list_of_results[k][2][countt][:, 1], linestyle="-", marker="o", label=list_of_results[k][0])

        if (broken == len(list_of_results)):
            plt.close()
            continue

        plt.xscale("log")
        plt.legend(loc="best")
        PDF_NAME_1 = "All_Rho_" + CURR_BN + "_Theta_vs_nR_" + curr_title + ".pdf"
        #plt.savefig(("All_SC_L_Theta_vs_nR.pdf"))
        plt.savefig((PDF_NAME_1))
        plt.close()



    for k in range(len(list_of_results)):
        plt.figure()
        curr_title = list_of_results[k][0] + ", BF = " + str(WBF)
        plt.title(curr_title)
        plt.ylabel("Theta")
        plt.xlabel("nR")

        for countt, aspect in enumerate(list_of_results[k][1]): #SCL_Labels
            try:
                chh = list_of_results[k][2][countt]
            except Exception:
                continue

            plt.plot(list_of_results[k][2][countt][:, 0], list_of_results[k][2][countt][:, 1], linestyle="-", marker="o", label=aspect)

        plt.xscale("log")
        plt.legend(loc="best")
        PDF_NAME_1 = "All_SCL_" + CURR_BN +  "_Theta_vs_nR_" + curr_title + ".pdf"
        #plt.savefig(("All_SC_L_Theta_vs_nR.pdf"))
        plt.savefig((PDF_NAME_1))
        plt.close()

    return list_of_results

###########################################################################################################################################################

def Plot_for_all_BN(Wpatch, WBF):

    dir_names = os.listdir()
    list_of_results = []
    blobN_label = ["Blob_1", "Blob_3", "Blob_5"]
    # blobN_label = ["Blob_5"]

    for bn in blobN_label:
    # for j in dir_names:
        #if (dir_names[:4] == "SC_L"):
        if bn not in dir_names:
            continue

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

    # SCL_label_list = ["SC_L_0", "SC_L_3", "SC_L_6", "SC_L_9", "SC_L_12", "SC_L_15"]
    SCL_label_list = ["SC_L_0", "SC_L_3", "SC_L_6"]

    # for countt, aspect in enumerate(list_of_results[0][1][0][1]): #SCL_LABELS
    for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
        plt.figure()
        curr_title = str(aspect) + ", BF = " + str(WBF)
        plt.title(curr_title)
        plt.ylabel("Theta")
        plt.xlabel("nR")

        for k in range(len(list_of_results)): #Blob number
            for rr in range(len(list_of_results[k][1])): # rho labels
                try:
                    this_rho = list_of_results[k][1][rr][0]
                    chh = list_of_results[k][1][rr][2][countt]
                    lbl = str(list_of_results[k][0]) + "_" + str(this_rho)
                except Exception:
                    continue

                plt.plot(list_of_results[k][1][rr][2][countt][:, 0], list_of_results[k][1][rr][2][countt][:, 1], linestyle="-", marker="o", label=lbl)

        plt.xscale("log")
        plt.legend(loc="best")
        PDF_NAME_1 = "All_BN_and_Rho_Theta_vs_nR_" + curr_title + ".pdf"
        #plt.savefig(("All_SC_L_Theta_vs_nR.pdf"))
        plt.savefig((PDF_NAME_1))
        plt.close()

    Rho_label_list = ["rho_1e4", "rho_1e5", "rho_1e6"]

    ##############################################################################################################

    for rr in range(len(Rho_label_list)): #rho labels
        plt.figure()
        curr_title = str(rr) + ", BF = " + str(WBF)
        plt.title(curr_title)
        plt.ylabel("Theta")
        plt.xlabel("nR")

        for k in range(len(list_of_results)): #Blob number
            for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
                try:
                    chh = list_of_results[k][1][rr][2][countt]
                    lbl = str(list_of_results[k][0]) + "_" + str(aspect)
                except Exception:
                    continue

                plt.plot(list_of_results[k][1][rr][2][countt][:, 0], list_of_results[k][1][rr][2][countt][:, 1], linestyle="-", marker="o", label=lbl)

        plt.xscale("log")
        plt.legend(loc="best")
        PDF_NAME_1 = "All_BN_and_SCL_Theta_vs_nR_" + curr_title + ".pdf"
        #plt.savefig(("All_SC_L_Theta_vs_nR.pdf"))
        plt.savefig((PDF_NAME_1))
        plt.close()


    ##############################################################################################################

    for k in range(len(list_of_results)):
        plt.figure()
        curr_title = str(list_of_results[k][0]) + ", BF = " + str(WBF)
        plt.title(curr_title)
        plt.ylabel("Theta")
        plt.xlabel("nR")

        for rr in range(len(Rho_label_list)): # rho labels
            for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
                try:
                    this_rho = list_of_results[k][1][rr][0]
                    chh = list_of_results[k][1][rr][2][countt]
                    lbl = str(this_rho) + "_" + str(aspect)
                except Exception:
                    continue

                plt.plot(list_of_results[k][1][rr][2][countt][:, 0], list_of_results[k][1][rr][2][countt][:, 1], linestyle="-", marker="o", label=lbl)

        plt.xscale("log")
        plt.legend(loc="best")
        PDF_NAME_1 = "All_Rho_and_SCL_Theta_vs_nR_" + curr_title + ".pdf"
        #plt.savefig(("All_SC_L_Theta_vs_nR.pdf"))
        plt.savefig((PDF_NAME_1))
        plt.close()


    return list_of_results

##############################################################################

def Plot_for_all_BF(Have_Patch):

    dir_names = os.listdir()
    BF_x_results = []

    for j in ["BF_5", "BF_4", "BF_3", "BF_2"]:
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

    print("\nSaving final array to file! Now you can redo the final part of the plotting without all the rest of the process!\n")
    np.save("Final_array.nump", FINAL_RESULT_LIST)

    Just_Plot()




#     blobN_label_list = ["Blob_1", "Blob_3", "Blob_5"]
#     Rho_label_list = ["rho_1e4", "rho_1e5", "rho_1e6"]
#     # SCL_label_list = ["SC_L_0", "SC_L_3", "SC_L_6", "SC_L_9", "SC_L_12", "SC_L_15"]
#     SCL_label_list = ["SC_L_0", "SC_L_3", "SC_L_6"]
#     BF_label_list = ["BF_4", "BF_3", "BF_2"]
#
#     #Plot comparison of Patch and No patch on same graph.. various combo
#
#     #Plotting for each triad Bonding energy / fugacity / Number of Blobs
#     for l, BFl in enumerate(BF_label_list):
#         for h, BLNh in enumerate(New[0][1][l][1]):
#             for p, rhop in enumerate(New[0][1][l][1][h][1]):
#
#                 plt.figure()
#                 curr_title = str(BLNh[0]) + " Patch vs No Patch, BF " + str(New[0][1][l][0]) + ", " + str(rhop[0])
#                 curr_file = str(BLNh[0]) + "_Patch_No_Patch_Theta_vs_nR_BF_" + str(New[0][1][l][0]) + "_" + str(rhop[0]) + ".pdf"
#                 plt.title(curr_title)
#                 plt.ylabel("Theta")
#                 plt.xlabel("nR")
#
#                 #Setting up colour cycle
#                 color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
#                 for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
#                     try:
#                         to_plot_NP = New[0][1][l][1][h][1][p][2][countt]
# #                        print("\nto_plot_NP : ")
#                         # print(to_plot_NP)
#                         to_plot_PA = New[1][1][l][1][h][1][p][2][countt]
# #                        print("\nto_plot_PA : ")
#                         # print(to_plot_PA)
#
#                     except Exception:
#                         print("Exception: l, h, p, count = " + str(l) + ", " + str(h) + ", " + str(p) + ", " + str(countt))
#                         continue
#
#                     label_1 = "NP, " + New[0][1][l][1][h][1][p][1][countt]
#                     label_2 = "PA, " + New[0][1][l][1][h][1][p][1][countt]
#                     print("label_1 = " + str(label_1))
#                     print("label_2 = " + str(label_2))
#
#                     plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[countt])
#                     plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[countt])
#
#
#                     plt.xscale("log")
#                     plt.legend(loc="best")
#                     plt.savefig((curr_file))
#                     plt.close()
#
#     #Plotting for each couple Bonding energy / fugacity
#     for l, BFl in enumerate(BF_label_list):
#         for p, rhop in enumerate(Rho_label_list):
#
#             plt.figure()
#             curr_title = "All Patch vs No Patch, " + str(New[0][1][l][0]) + ", " + str(rhop)
#             curr_file = "All_Patch_No_Patch_Theta_vs_nR_" + str(New[0][1][l][0]) + "_" + str(rhop) + ".pdf"
#             plt.title(curr_title)
#             plt.ylabel("Theta")
#             plt.xlabel("nR")
#
#             #Setting up colour cycle
#             color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
#             for h, BLNh in enumerate(New[0][1][l][1]):
#                 for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
#                     try:
#                         print("\n\nl, BFl = " + str(l) + ", " + str(New[0][1][l][0]))
#                         print("p, rhop = " + str(p) + ", " + str(rhop))
#                         print("h, BLNh = " + str(h) + ", " + str(BLNh[0]))
#                         # print("countt, aspect = " + str(countt) + ", " + str(aspect))
#                         to_plot_NP = New[0][1][l][1][h][1][p][2][countt]
#                         to_plot_PA = New[1][1][l][1][h][1][p][2][countt]
#                         # list_of_results[ ][ ][ ][ ][k][1][rr][2][countt]
#                         # print("to_plot_NP: ")
#                         # print(to_plot_NP)
#                         # print("to_plot_PA: ")
#                         # print(to_plot_PA)
#
#                     except Exception:
#                         print("Exception!!")
#                         continue
#
#                     label_1 = "NP, " + str(BLNh[0]) + ", " + str(aspect)
#                     label_2 = "PA, " + str(BLNh[0]) + ", " + str(aspect)
#                     color_count = h + countt
#                     print("label_1 = " + str(label_1))
#                     print("label_2 = " + str(label_2))
#
#
#                     plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[color_count])
#                     plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[color_count])
#
#
#             plt.xscale("log")
#             plt.legend(loc="best")
#             plt.savefig((curr_file))
#             plt.close()
#
#
#     #Plotting for each triad Bonding energy / fugacity / SCL
#     for l, BFl in enumerate(BF_label_list):
#         for p, rhop in enumerate(Rho_label_list):
#             for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
#                 # New[0][1][l][1][0] = BLNh
#                 # New[0][1][l][0] = BFl
#
#                 plt.figure()
#                 curr_title = str(aspect) + " Patch vs No Patch, " + str(New[0][1][l][0]) + ", " + str(rhop)
#                 curr_file = str(aspect) + "_Patch_No_Patch_Theta_vs_nR_" + str(New[0][1][l][0]) + "_" + str(rhop) + ".pdf"
#                 plt.title(curr_title)
#                 plt.ylabel("Theta")
#                 plt.xlabel("nR")
#
#                 #Setting up colour cycle
#                 color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
#                 for h, BLNh in enumerate(New[0][1][l][1]):
#                     try:
#                         to_plot_NP = New[0][1][l][1][h][1][p][2][countt]
#                         to_plot_PA = New[1][1][l][1][h][1][p][2][countt]
#                     except Exception:
#                         continue
#
#                     label_1 = "NP, " + str(BLNh[0])
#                     label_2 = "PA, " + str(BLNh[0])
#
#                     plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[h])
#                     plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[h])
#
#
#                 plt.xscale("log")
#                 plt.legend(loc="best")
#                 plt.savefig((curr_file))
#                 plt.close()
#
#     #############################################################################################
#
#     ##Entering new directory for less important graphs
#     os.makedirs('Rho_BF_Changes', exist_ok=True)
#     os.chdir('Rho_BF_Changes')
#
#     #Varying Binding energy
#     for h, BLNh in enumerate(blobN_label_list):
#         for p, rhop in enumerate(Rho_label_list):
#             for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
#
#                 plt.figure()
#                 # New[0][1][l][1][0] = BLNh
#                 # New[0][1][l][0] = BFl
#
#                 curr_title = str(aspect) + " Patch vs No Patch, " + str(New[0][1][l][1][h][0]) + ", " + str(rhop)
#                 curr_file = str(aspect) + "_Patch_No_Patch_Theta_vs_nR_" + str(New[0][1][l][1][h][0]) + "_" + str(rhop) + ".pdf"
#                 plt.title(curr_title)
#                 plt.ylabel("Theta")
#                 plt.xlabel("nR")
#
#                 #Setting up colour cycle
#                 color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
#                 for l, BFl in enumerate(BF_label_list):
#                     try:
#                         to_plot_NP = New[0][1][l][1][h][1][p][2][countt]
#                         to_plot_PA = New[1][1][l][1][h][1][p][2][countt]
#                     except Exception:
#                         continue
#
#                     label_1 = "NP, " + str(New[0][1][l][0])
#                     label_2 = "PA, " + str(New[0][1][l][0])
#
#                     plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[l])
#                     plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[l])
#
#
#                 plt.xscale("log")
#                 plt.legend(loc="best")
#                 plt.savefig((curr_file))
#                 plt.close()
#
#     #Varying fugacity
#     for h, BLNh in enumerate(blobN_label_list):
#         for l, BFl in enumerate(BF_label_list):
#             for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
#
#                 plt.figure()
#                 curr_title = str(aspect) + " Patch vs No Patch, " + str(New[0][1][l][1][h][0]) + ", " + str(New[0][1][l][0])
#                 curr_file = str(aspect) + "_Patch_No_Patch_Theta_vs_nR_" + str(New[0][1][l][1][h][0]) + "_" + str(New[0][1][l][0]) + ".pdf"
#                 plt.title(curr_title)
#                 plt.ylabel("Theta")
#                 plt.xlabel("nR")
#
#                 #Setting up colour cycle
#                 color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
#                 for p, rhop in enumerate(Rho_label_list):
#                     try:
#                         to_plot_NP = New[0][1][l][1][h][1][p][2][countt]
#                         to_plot_PA = New[1][1][l][1][h][1][p][2][countt]
#                     except Exception:
#                         continue
#
#                     label_1 = "NP, " + str(rhop)
#                     label_2 = "PA, " + str(rhop)
#
#                     plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[p])
#                     plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[p])
#
#
#                 plt.xscale("log")
#                 plt.legend(loc="best")
#                 plt.savefig((curr_file))
#                 plt.close()
#
#
#     os.chdir('../')
#
#     Plot_selectivity_PaNoPa(FINAL_RESULT_LIST)

    # os.makedirs('Superselectivity_plots', exist_ok=True)
    # os.chdir('Superselectivity_plots')
    # #Selectivity plot function
    # os.chdir('../')

    # os.makedirs('Colloid_alignment_plots', exist_ok=True)
    # os.chdir('Colloid_alignment_plots')
    # #Colloids alignment plot function
    # os.chdir('../')

    #############################################################################################

    return

##############################################################################

def Plot_selectivity_PaNoPa(RES_ARRAY):

    os.makedirs('Superselectivity_results', exist_ok=True)
    os.chdir('Superselectivity_results')
    File_list = os.listdir()

    for fl in File_list:
        if fl.endswith(".pdf"):
            os.remove(fl)


    blobN_label_list = ["Blob_1", "Blob_3", "Blob_5"]
    Rho_label_list = ["rho_1e4", "rho_1e5", "rho_1e6"]
    SCL_label_list = ["SC_L_0", "SC_L_3", "SC_L_6", "SC_L_9", "SC_L_12", "SC_L_15"]
    CLEAN_SCL_label_list = [0, 3, 6, 9, 12, 15]
    # SCL_label_list = ["SC_L_0", "SC_L_3", "SC_L_6"]
    # BF_label_list = ["BF_5", "BF_4", "BF_3", "BF_2"]
    BF_label_list = ["BF_5", "BF_4", "BF_3"]


    #No patch loop
    NP_max_arr = []
    NP_maxpos_arr = []
    NP_maxTheta_arr = []

    for bfree in range(len(RES_ARRAY[0][1])): #binding free energy
        NP_max_BF = []
        NP_maxpos_BF = []
        NP_maxTheta_BF = []

        for bnk in range(len(RES_ARRAY[0][1][bfree][1])): #Blob per colloid
            NP_max_BN = []
            NP_maxpos_BN = []
            NP_maxTheta_BN = []

            for r in range(len(RES_ARRAY[0][1][bfree][1][bnk][1])): #fugacity
                NP_max_rho = []
                NP_maxpos_rho = []
                NP_maxTheta_rho = []

                for SCL in range(len(RES_ARRAY[0][1][bfree][1][bnk][1][r][2])): #fugacity
                    # to_update = np.log(RES_ARRAY[0][1][bfree][1][bnk][1][r][2][SCL])
                    to_update = RES_ARRAY[0][1][bfree][1][bnk][1][r][2][SCL]

                    MAXIMUM_ADSORPTION_POS = to_update[:,1].argmax() #Position of maximum adsorption
                    MAXIMUM_ADSORPTION = to_update[:,1].max() #Position of maximum adsorption

                    to_update = np.log(to_update)
                    New_arr = to_update.copy()

                    for ll in range(len(New_arr) - 2):
                        Delta_y = to_update[ll+2, 1] - to_update[ll, 1]
                        Delta_x = to_update[ll+2, 0] - to_update[ll, 0]

                        Alpha_ll = Delta_y / Delta_x
                        New_arr[ll + 1, 1] = Alpha_ll

                    try:
                        New_arr[0, 1] = New_arr[1, 1]
                        New_arr[-1, 1] = New_arr[-2, 1]
                    except Exception:
                        print("Exception")

                    # New_arr = New_arr[New_arr[:,0] <= MAXIMUM_ADSORPTION] #Removing values of selectivity after adsorption maximum
                    # New_arr = New_arr[New_arr[:,1] > 0] #Removing all negative selectivity

                    RES_ARRAY[0][1][bfree][1][bnk][1][r][2][SCL][:, 1] = New_arr[:, 1]

                    NP_max_rho.append(New_arr[:MAXIMUM_ADSORPTION_POS, 1].max())
                    NP_maxpos_rho.append(New_arr[:MAXIMUM_ADSORPTION_POS, 1].argmax())
                    NP_maxTheta_rho.append(MAXIMUM_ADSORPTION)

                NP_max_BN.append(NP_max_rho)
                NP_maxpos_BN.append(NP_maxpos_rho)
                NP_maxTheta_BN.append(NP_maxTheta_rho)

            NP_max_BF.append(NP_max_BN)
            NP_maxpos_BF.append(NP_maxpos_BN)
            NP_maxTheta_BF.append(NP_maxTheta_BN)

        NP_max_arr.append(NP_max_BF)
        NP_maxpos_arr.append(NP_maxpos_BF)
        NP_maxTheta_arr.append(NP_maxTheta_BF)


    #Patch loop
    PA_max_arr = []
    PA_maxpos_arr = []
    PA_maxTheta_arr = []

    for bfree in range(len(RES_ARRAY[1][1])): #binding free energy
        PA_max_BF = []
        PA_maxpos_BF = []
        PA_maxTheta_BF = []

        for bnk in range(len(RES_ARRAY[1][1][bfree][1])): #Blob per colloid
            PA_max_BN = []
            PA_maxpos_BN = []
            PA_maxTheta_BN = []

            for r in range(len(RES_ARRAY[1][1][bfree][1][bnk][1])): #fugacity
                PA_max_rho = []
                PA_maxpos_rho = []
                PA_maxTheta_rho = []

                for SCL in range(len(RES_ARRAY[1][1][bfree][1][bnk][1][r][2])): #fugacity
                    # to_update = np.log(RES_ARRAY[1][1][bfree][1][bnk][1][r][2][SCL])
                    to_update = RES_ARRAY[1][1][bfree][1][bnk][1][r][2][SCL]
                    MAXIMUM_ADSORPTION_POS = to_update[:,1].argmax() #Position of maximum adsorption
                    MAXIMUM_ADSORPTION = to_update[:,1].max() #Position of maximum adsorption

                    to_update = np.log(to_update)
                    New_arr = to_update.copy()
                    # New_arr = to_update[1:-1]

                    for ll in range(len(New_arr) - 2):
                        Delta_y = to_update[ll+2, 1] - to_update[ll, 1]
                        Delta_x = to_update[ll+2, 0] - to_update[ll, 0]

                        Alpha_ll = Delta_y / Delta_x
                        New_arr[ll + 1, 1] = Alpha_ll

                    # New_arr = New_arr[New_arr[:,0] <= MAXIMUM_ADSORPTION] #Removing values of selectivity after adsorption maximum
                    # New_arr = New_arr[New_arr[:,1] > 0] #Removing all negative selectivity
                    try:
                        New_arr[0, 1] = New_arr[1, 1]
                        New_arr[-1, 1] = New_arr[-2, 1]
                    except Exception:
                        print("Exception")

                    RES_ARRAY[1][1][bfree][1][bnk][1][r][2][SCL][:, 1] = New_arr[:, 1]

                    PA_max_rho.append(New_arr[:MAXIMUM_ADSORPTION_POS, 1].max())
                    PA_maxpos_rho.append(New_arr[:MAXIMUM_ADSORPTION_POS, 1].argmax())
                    PA_maxTheta_rho.append(MAXIMUM_ADSORPTION)

                PA_max_BN.append(PA_max_rho)
                PA_maxpos_BN.append(PA_maxpos_rho)
                PA_maxTheta_BN.append(PA_maxTheta_rho)

            PA_max_BF.append(PA_max_BN)
            PA_maxpos_BF.append(PA_maxpos_BN)
            PA_maxTheta_BF.append(PA_maxTheta_BN)

        PA_max_arr.append(PA_max_BF)
        PA_maxpos_arr.append(PA_maxpos_BF)
        PA_maxTheta_arr.append(PA_maxTheta_BF)

    #Plotting for each couple Bonding energy / fugacity
    for l, BFl in enumerate(RES_ARRAY[0][1]):
        for p, rhop in enumerate(Rho_label_list):

            plt.figure()
            curr_title = "BF = -" + str(BFl[0][-1]) + ", rho = 1e-" + str(rhop[-1])
            curr_file = "nR_vs_Alpha_BF_" + str(BFl[0][-1]) + "_rho_" + str(rhop[-3:]) + ".pdf"
            plt.title(curr_title)
            plt.ylabel("Alpha")
            plt.xlabel("nR")

            #Setting up colour cycle
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            for h, BLNh in enumerate(RES_ARRAY[0][1][l][1]):
                for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
                    try:
                        to_plot_NP = RES_ARRAY[0][1][l][1][h][1][p][2][countt]
                        to_plot_PA = RES_ARRAY[1][1][l][1][h][1][p][2][countt]
                    except Exception:
                        continue

                    if len(BLNh[0]) == 6:
                        BLNh_clean = BLNh[0][-1]
                    else:
                        BLNh_clean = BLNh[0][-2:]

                    if len(aspect) == 6:
                        SCL_clean = aspect[-1]
                    else:
                        SCL_clean = aspect[-2:]

                    label_1 = "NP, Blob=" + str(BLNh_clean) + ", SCL=" + str(SCL_clean)
                    label_2 = "PA, Blob=" + str(BLNh_clean) + ", SCL=" + str(SCL_clean)
                    # color_count = h + countt
                    color_count = countt

                    plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[color_count])
                    plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[color_count])


            plt.xscale("log")
            plt.legend(loc="best")
            plt.savefig((curr_file))
            plt.close()

    #Plotting for each triad Bonding energy / fugacity / Number of Blobs
    for l, BFl in enumerate(RES_ARRAY[0][1]):
        for h, BLNh in enumerate(BFl[1]):
            for p, rhop in enumerate(BLNh[1]):

                if len(BLNh[0]) == 6:
                    BLNh_clean = BLNh[0][-1]
                else:
                    BLNh_clean = BLNh[0][-2:]

                plt.figure()
                curr_title = "Blob " + str(BLNh_clean) + ", BF = -" + str(BFl[0][-1]) + ", rho = 1e-" + str(rhop[0][-1])
                curr_file = "nR_vs_Alpha_Blob_" + str(BLNh_clean) + "_BF_" + str(BFl[0][-1]) + "_rho_" + str(rhop[0][-3:]) + ".pdf"
                plt.title(curr_title)
                plt.ylabel("Alpha")
                plt.xlabel("nR")

                #Setting up colour cycle
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

                for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
                    try:
                        to_plot_NP = RES_ARRAY[0][1][l][1][h][1][p][2][countt]
                        # print("\nto_plot_NP : ")
                        # print(to_plot_NP)
                        to_plot_PA = RES_ARRAY[1][1][l][1][h][1][p][2][countt]
                    except Exception:
                        continue

                    if len(aspect) == 6:
                        SCL_clean = aspect[-1]
                    else:
                        SCL_clean = aspect[-2:]

                    label_1 = "NP, SCL=" + str(SCL_clean)
                    label_2 = "PA, SCL=" + str(SCL_clean)

                    plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[countt])
                    plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[countt])


                plt.xscale("log")
                plt.legend(loc="best")
                plt.savefig((curr_file))
                plt.close()

    #Plotting for each triad Bonding energy / fugacity / SCL
    for l, BFl in enumerate(RES_ARRAY[0][1]):
        for p, rhop in enumerate(Rho_label_list):
            for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS

                if len(aspect) == 6:
                    SCL_clean = aspect[-1]
                else:
                    SCL_clean = aspect[-2:]

                plt.figure()
                curr_title = "BF = - " + str(BFl[0][-1]) + ", rho = 1e-" + str(rhop[-1]) + ", SCL = " + str(SCL_clean)
                curr_file = "nR_vs_Alpha_BF_" + str(BFl[0][-1]) + "_rho_" + str(rhop[-3:]) + "_SCL_" + str(SCL_clean) + ".pdf"
                plt.title(curr_title)
                plt.ylabel("Alpha")
                plt.xlabel("nR")

                #Setting up colour cycle
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

                for h, BLNh in enumerate(RES_ARRAY[0][1][l][1]):
                    try:
                        to_plot_NP = RES_ARRAY[0][1][l][1][h][1][p][2][countt]
                        to_plot_PA = RES_ARRAY[1][1][l][1][h][1][p][2][countt]
                    except Exception:
                        continue

                    if len(BLNh[0]) == 6:
                        BLNh_clean = BLNh[0][-1]
                    else:
                        BLNh_clean = BLNh[0][-2:]

                    label_1 = "NP, Blob=" + str(BLNh_clean)
                    label_2 = "PA, Blob=" + str(BLNh_clean)

                    plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[h])
                    plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[h])


                plt.xscale("log")
                plt.legend(loc="best")
                plt.savefig((curr_file))
                plt.close()

    os.chdir('../')


    ###############################################################
    ###############################################################
    #Plotting maximum Alpha and maximum Alpha pos
    print("\nPlotting maximum Alpha and maximum Alpha pos ! \n ")
    os.makedirs('Max_Alpha', exist_ok=True)
    os.chdir('Max_Alpha')
    File_list = os.listdir()

    for fl in File_list:
        if fl.endswith(".pdf"):
            os.remove(fl)

    #################
    ### ALPHA MAX ###
    #Couple BF / Blob_N
    for l, BFl in enumerate(RES_ARRAY[0][1]):
        # for h, BLNh in enumerate(New[0][1][l][1]):
        for h, BLNh in enumerate(BFl[1]):
            if len(BLNh[0]) == 6:
                BLNh_clean = BLNh[0][-1]
            else:
                BLNh_clean = BLNh[0][-2:]

            plt.figure()
            curr_title = "Blob " + str(BLNh_clean) + ", BF = -" + str(BFl[0][-1])
            curr_file = "AlphaMax_vs_SCL_Blob_" + str(BLNh_clean) + "_BF_" + str(BFl[0][-1]) + ".pdf"
            plt.title(curr_title)
            plt.ylabel("Alpha Max")
            plt.xlabel("SCL")

            #Setting up colour cycle
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # for p, rhop in enumerate(New[0][1][l][1][h][1]):
            for p, rhop in enumerate(BLNh[1]):
                label_1 = "NP, rho=1e-" + str(rhop[0][-1])
                label_2 = "PA, rho=1e-" + str(rhop[0][-1:])
                # print("label_1 = " + str(label_1))
                # print("label_2 = " + str(label_2))
                try:
                    plt.plot(CLEAN_SCL_label_list, NP_max_arr[l][h][p], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[p])
                    plt.plot(CLEAN_SCL_label_list, PA_max_arr[l][h][p], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[p])

                except Exception:
                    continue

            plt.legend(loc="best")
            plt.savefig((curr_file))
            plt.close()

    #Couple BF / Rho
    for l, BFl in enumerate(RES_ARRAY[0][1]):
        for p, rhop in enumerate(Rho_label_list):

            plt.figure()
            curr_title = "BF = -" + str(BFl[0][-1]) + ", rho = 1e-" + str(rhop[-1])
            curr_file = "AlphaMax_vs_SCL_BF_" + str(BFl[0][-1]) + "_rho_" + str(rhop[-3:]) + ".pdf"
            plt.title(curr_title)
            plt.ylabel("Alpha Max")
            plt.xlabel("SCL")

            #Setting up colour cycle
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # for h, BLNh in enumerate(New[0][1][l][1]):
            for h, BLNh in enumerate(BFl[1]):
                if len(BLNh[0]) == 6:
                    BLNh_clean = BLNh[0][-1]
                else:
                    BLNh_clean = BLNh[0][-2:]

                label_1 = "NP, Blob=" + str(BLNh_clean)
                label_2 = "PA, Blob=" + str(BLNh_clean)
                # print("label_1 = " + str(label_1))
                # print("label_2 = " + str(label_2))
                try:
                    plt.plot(CLEAN_SCL_label_list, NP_max_arr[l][h][p], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[h])
                    plt.plot(CLEAN_SCL_label_list, PA_max_arr[l][h][p], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[h])

                except Exception:
                    continue


            plt.legend(loc="best")
            plt.savefig((curr_file))
            plt.close()


    #################
    ### ALPHA MAX POS ###
    #Couple BF / Blob_N
    for l, BFl in enumerate(RES_ARRAY[0][1]):
        # for h, BLNh in enumerate(New[0][1][l][1]):
        for h, BLNh in enumerate(BFl[1]):
            if len(BLNh[0]) == 6:
                BLNh_clean = BLNh[0][-1]
            else:
                BLNh_clean = BLNh[0][-2:]

            plt.figure()
            curr_title = "Blob " + str(BLNh_clean) + ", BF = -" + str(BFl[0][-1])
            curr_file = "AlphaPOS_Max_vs_SCL_Blob_" + str(BLNh_clean) + "_BF_" + str(BFl[0][-1]) + ".pdf"
            plt.title(curr_title)
            plt.ylabel("Alpha Max nR")
            plt.xlabel("SCL")

            #Setting up colour cycle
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # for p, rhop in enumerate(New[0][1][l][1][h][1]):
            for p, rhop in enumerate(BLNh[1]):
                label_1 = "NP, rho=1e-" + str(rhop[0][-1])
                label_2 = "PA, rho=1e-" + str(rhop[0][-1:])
                # print("label_1 = " + str(label_1))
                # print("label_2 = " + str(label_2))
                try:
                    plt.plot(CLEAN_SCL_label_list, NP_maxpos_arr[l][h][p], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[p])
                    plt.plot(CLEAN_SCL_label_list, PA_maxpos_arr[l][h][p], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[p])

                except Exception:
                    continue

            plt.legend(loc="best")
            plt.savefig((curr_file))
            plt.close()

    #Couple BF / Rho
    for l, BFl in enumerate(RES_ARRAY[0][1]):
        for p, rhop in enumerate(Rho_label_list):

            plt.figure()
            curr_title = "BF = -" + str(BFl[0][-1]) + ", rho = 1e-" + str(rhop[-1])
            curr_file = "AlphaPOS_Max_vs_SCL_BF_" + str(BFl[0][-1]) + "_rho_" + str(rhop[-3:]) + ".pdf"
            plt.title(curr_title)
            plt.ylabel("Alpha Max nR")
            plt.xlabel("SCL")

            #Setting up colour cycle
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # for h, BLNh in enumerate(New[0][1][l][1]):
            for h, BLNh in enumerate(BFl[1]):
                if len(BLNh[0]) == 6:
                    BLNh_clean = BLNh[0][-1]
                else:
                    BLNh_clean = BLNh[0][-2:]

                label_1 = "NP, Blob=" + str(BLNh_clean)
                label_2 = "PA, Blob=" + str(BLNh_clean)
                # print("label_1 = " + str(label_1))
                # print("label_2 = " + str(label_2))
                try:
                    plt.plot(CLEAN_SCL_label_list, NP_maxpos_arr[l][h][p], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[h])
                    plt.plot(CLEAN_SCL_label_list, PA_maxpos_arr[l][h][p], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[h])

                except Exception:
                    continue


            plt.legend(loc="best")
            plt.savefig((curr_file))
            plt.close()

    os.chdir('../')

    #################
    ### THETA MAX ###
    print("\nPlotting maximum Theta ! \n ")
    os.makedirs('Max_Theta', exist_ok=True)
    os.chdir('Max_Theta')
    File_list = os.listdir()

    for fl in File_list:
        if fl.endswith(".pdf"):
            os.remove(fl)

    #Couple BF / Blob_N
    for l, BFl in enumerate(RES_ARRAY[0][1]):
        # for h, BLNh in enumerate(New[0][1][l][1]):
        for h, BLNh in enumerate(BFl[1]):
            if len(BLNh[0]) == 6:
                BLNh_clean = BLNh[0][-1]
            else:
                BLNh_clean = BLNh[0][-2:]

            plt.figure()
            curr_title = "Blob " + str(BLNh_clean) + ", BF = -" + str(BFl[0][-1])
            curr_file = "Theta_Max_vs_SCL_Blob_" + str(BLNh_clean) + "_BF_" + str(BFl[0][-1]) + ".pdf"
            plt.title(curr_title)
            plt.ylabel("Theta Max")
            plt.xlabel("SCL")

            #Setting up colour cycle
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # for p, rhop in enumerate(New[0][1][l][1][h][1]):
            for p, rhop in enumerate(BLNh[1]):
                label_1 = "NP, rho=1e-" + str(rhop[0][-1])
                label_2 = "PA, rho=1e-" + str(rhop[0][-1:])
                # print("label_1 = " + str(label_1))
                # print("label_2 = " + str(label_2))
                try:
                    plt.plot(CLEAN_SCL_label_list, NP_maxTheta_arr[l][h][p], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[p])
                    plt.plot(CLEAN_SCL_label_list, PA_maxTheta_arr[l][h][p], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[p])

                except Exception:
                    continue

            plt.legend(loc="best")
            plt.savefig((curr_file))
            plt.close()

    #Couple BF / Rho
    for l, BFl in enumerate(RES_ARRAY[0][1]):
        for p, rhop in enumerate(Rho_label_list):

            plt.figure()
            curr_title = "BF = -" + str(BFl[0][-1]) + ", rho = 1e-" + str(rhop[-1])
            curr_file = "Theta_Max_vs_SCL_BF_" + str(BFl[0][-1]) + "_rho_" + str(rhop[-3:]) + ".pdf"
            plt.title(curr_title)
            plt.ylabel("Theta Max")
            plt.xlabel("SCL")

            #Setting up colour cycle
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # for h, BLNh in enumerate(New[0][1][l][1]):
            for h, BLNh in enumerate(BFl[1]):
                if len(BLNh[0]) == 6:
                    BLNh_clean = BLNh[0][-1]
                else:
                    BLNh_clean = BLNh[0][-2:]

                label_1 = "NP, Blob=" + str(BLNh_clean)
                label_2 = "PA, Blob=" + str(BLNh_clean)
                # print("label_1 = " + str(label_1))
                # print("label_2 = " + str(label_2))
                try:
                    plt.plot(CLEAN_SCL_label_list, NP_maxTheta_arr[l][h][p], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[h])
                    plt.plot(CLEAN_SCL_label_list, PA_maxTheta_arr[l][h][p], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[h])

                except Exception:
                    continue


            plt.legend(loc="best")
            plt.savefig((curr_file))
            plt.close()




    return

##############################################################################

def Just_Plot():

    File_list = os.listdir()

    for fl in File_list:
        if fl.endswith(".pdf"):
            os.remove(fl)

    New = np.load("Final_array.nump.npy", allow_pickle=True)
    # print("New:")
    # print(New)

    blobN_label_list = ["Blob_1", "Blob_3", "Blob_5"]
    Rho_label_list = ["rho_1e4", "rho_1e5", "rho_1e6"]
    SCL_label_list = ["SC_L_0", "SC_L_3", "SC_L_6", "SC_L_9", "SC_L_12", "SC_L_15"]
    # SCL_label_list = ["SC_L_0", "SC_L_3", "SC_L_6"]
    # BF_label_list = ["BF_5", "BF_4", "BF_3", "BF_2"]
    BF_label_list = ["BF_5", "BF_4", "BF_3"]

    #Plot comparison of Patch and No patch on same graph.. various combo

    #Plotting for each triad Bonding energy / fugacity / Number of Blobs
    for l, BFl in enumerate(New[0][1]):
        # for h, BLNh in enumerate(New[0][1][l][1]):
        for h, BLNh in enumerate(BFl[1]):
            # for p, rhop in enumerate(New[0][1][l][1][h][1]):
            for p, rhop in enumerate(BLNh[1]):

                if len(BLNh[0]) == 6:
                    BLNh_clean = BLNh[0][-1]
                else:
                    BLNh_clean = BLNh[0][-2:]

                plt.figure()
                curr_title = "Blob " + str(BLNh_clean) + ", BF = -" + str(BFl[0][-1]) + ", rho = 1e-" + str(rhop[0][-1])
                curr_file = "nR_vs_Theta_Blob_" + str(BLNh_clean) + "_BF_" + str(BFl[0][-1]) + "_rho_" + str(rhop[0][-3:]) + ".pdf"
                plt.title(curr_title)
                plt.ylabel("Theta")
                plt.xlabel("nR")

                #Setting up colour cycle
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

                for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
                    try:
                        to_plot_NP = New[0][1][l][1][h][1][p][2][countt]
#                        print("\nto_plot_NP : ")
                        # print(to_plot_NP)
                        to_plot_PA = New[1][1][l][1][h][1][p][2][countt]
#                        print("\nto_plot_PA : ")
                        # print(to_plot_PA)

                    except Exception:
                        print("Exception: l, h, p, count = " + str(l) + ", " + str(h) + ", " + str(p) + ", " + str(countt))
                        continue

                    if len(New[0][1][l][1][h][1][p][1][countt]) == 6:
                        SCL_clean = New[0][1][l][1][h][1][p][1][countt][-1]
                    else:
                        SCL_clean = New[0][1][l][1][h][1][p][1][countt][-2:]

                    label_1 = "NP, SCL=" + str(SCL_clean)
                    label_2 = "PA, SCL=" + str(SCL_clean)
                    # print("label_1 = " + str(label_1))
                    # print("label_2 = " + str(label_2))

                    plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[countt])
                    plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[countt])


                plt.xscale("log")
                plt.legend(loc="best")
                plt.savefig((curr_file))
                plt.close()

    #Plotting for each couple Bonding energy / fugacity
    # for l, BFl in enumerate(BF_label_list):
    for l, BFl in enumerate(New[0][1]):
        for p, rhop in enumerate(Rho_label_list):

            plt.figure()
            curr_title = "BF = -" + str(BFl[0][-1]) + ", rho = 1e-" + str(rhop[-1])
            curr_file = "nR_vs_Theta_BF_" + str(BFl[0][-1]) + "_rho_" + str(rhop[-3:]) + ".pdf"
            plt.title(curr_title)
            plt.ylabel("Theta")
            plt.xlabel("nR")

            #Setting up colour cycle
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # for h, BLNh in enumerate(New[0][1][l][1]):
            for h, BLNh in enumerate(BFl[1]):
                for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
                    try:
                        # print("\n\nl, BFl = " + str(l) + ", " + str(New[0][1][l][0]))
                        # print("p, rhop = " + str(p) + ", " + str(rhop))
                        # print("h, BLNh = " + str(h) + ", " + str(BLNh[0]))
                        # # print("countt, aspect = " + str(countt) + ", " + str(aspect))
                        to_plot_NP = New[0][1][l][1][h][1][p][2][countt]
                        to_plot_PA = New[1][1][l][1][h][1][p][2][countt]
                        # list_of_results[ ][ ][ ][ ][k][1][rr][2][countt]
                        # print("to_plot_NP: ")
                        # print(to_plot_NP)
                        # print("to_plot_PA: ")
                        # print(to_plot_PA)

                    except Exception:
                        print("Exception!!")
                        continue

                    if len(BLNh[0]) == 6:
                        BLNh_clean = BLNh[0][-1]
                    else:
                        BLNh_clean = BLNh[0][-2:]

                    if len(aspect) == 6:
                        SCL_clean = aspect[-1]
                    else:
                        SCL_clean = aspect[-2:]


                    label_1 = "NP, Blob=" + str(BLNh_clean) + ", SCL=" + str(SCL_clean)
                    label_2 = "PA, Blob=" + str(BLNh_clean) + ", SCL=" + str(SCL_clean)
                    # color_count = 6*h + countt
                    color_count = countt
                    # print("label_1 = " + str(label_1))
                    # print("label_2 = " + str(label_2))


                    plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[color_count])
                    plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[color_count])


            plt.xscale("log")
            plt.legend(loc="best")
            plt.savefig((curr_file))
            plt.close()


    #Plotting for each triad Bonding energy / fugacity / SCL
    # for l, BFl in enumerate(BF_label_list):
    for l, BFl in enumerate(New[0][1]):
        for p, rhop in enumerate(Rho_label_list):
            for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
                # New[0][1][l][1][0] = BLNh
                # New[0][1][l][0] = BFl

                if len(aspect) == 6:
                    SCL_clean = aspect[-1]
                else:
                    SCL_clean = aspect[-2:]


                plt.figure()
                curr_title = "BF = -" + str(BFl[0][-1]) + ", rho = 1e-" + str(rhop[-1]) + ", SCL = " + str(SCL_clean)
                curr_file = "nR_vs_Theta_BF_" + str(BFl[0][-1]) + "_rho_" + str(rhop[-3:]) + "_SCL_" + str(SCL_clean) + ".pdf"
                plt.title(curr_title)
                plt.ylabel("Theta")
                plt.xlabel("nR")

                #Setting up colour cycle
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

                # for h, BLNh in enumerate(New[0][1][l][1]):
                for h, BLNh in enumerate(BFl[1]):
                    try:
                        to_plot_NP = New[0][1][l][1][h][1][p][2][countt]
                        to_plot_PA = New[1][1][l][1][h][1][p][2][countt]
                    except Exception:
                        continue

                    if len(BLNh[0]) == 6:
                        BLNh_clean = BLNh[0][-1]
                    else:
                        BLNh_clean = BLNh[0][-2:]

                    label_1 = "NP, Blob=" + str(BLNh_clean)
                    label_2 = "PA, Blob=" + str(BLNh_clean)

                    plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[h])
                    plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[h])


                plt.xscale("log")
                plt.legend(loc="best")
                plt.savefig((curr_file))
                plt.close()

    #############################################################################################

    ##Entering new directory for less important graphs
    os.makedirs('Rho_BF_Changes', exist_ok=True)
    os.chdir('Rho_BF_Changes')
    File_list = os.listdir()

    for fl in File_list:
        if fl.endswith(".pdf"):
            os.remove(fl)

    print("\nPlotting Rho_BF_Changes ! \n ")

                    # New[0][1][l][1][h][0] = BLNh
                    # New[0][1][l][0] = BFl
    ##
    #Varying Binding energy
    #Plotting for each triad SCL / fugacity / Number of Blobs
    for h, BLNh in enumerate(blobN_label_list):
        for p, rhop in enumerate(Rho_label_list):
            for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
                try:
                    plt.figure()
                    # New[0][1][l][1][h][0] = BLNh
                    # New[0][1][l][0] = BFl
                    BLNh = New[0][1][0][1][h][0]
                    if len(BLNh) == 6:
                        BLNh_clean = BLNh[-1]
                    else:
                        BLNh_clean = BLNh[-2:]

                    if len(aspect) == 6:
                        SCL_clean = aspect[-1]
                    else:
                        SCL_clean = aspect[-2:]

                    curr_title = "Blob " + str(BLNh_clean) + ", rho = 1e-" + str(rhop[-1]) + ", SCL =" + str(SCL_clean)
                    curr_file = "nR_vs_Theta_Blob_" + str(BLNh_clean) + "_rho_" + str(rhop[-3:])+ "_SCL_" + str(SCL_clean) + ".pdf"
                    plt.title(curr_title)
                    plt.ylabel("Theta")
                    plt.xlabel("nR")

                except Exception:
                    continue

                #Setting up colour cycle
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

                for l, BFl in enumerate(BF_label_list):
                    try:
                        to_plot_NP = New[0][1][l][1][h][1][p][2][countt]
                        to_plot_PA = New[1][1][l][1][h][1][p][2][countt]
                    except Exception:
                        print("Exception: l, h, p, count = " + str(l) + ", " + str(h) + ", " + str(p) + ", " + str(countt))
                        continue

                    # label_1 = "NP, " + str(New[0][1][l][0])
                    # label_2 = "PA, " + str(New[0][1][l][0])
                    BFl = New[0][1][l][0]
                    label_1 = "NP, BF=-" + str(BFl[-1])
                    label_2 = "PA, BF=-" + str(BFl[-1])
                    print("label_1 = " + str(label_1))
                    print("label_2 = " + str(label_2))

                    plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[l])
                    plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[l])


                plt.xscale("log")
                plt.legend(loc="best")
                plt.savefig((curr_file))
                plt.close()

    #Plotting for each couple fugacity / Number of Blobs
    for h, BLNh in enumerate(blobN_label_list):
        for p, rhop in enumerate(Rho_label_list):
            try:
                BLNh = New[0][1][0][1][h][0]
            except Exception:
                continue

            if len(BLNh) == 6:
                BLNh_clean = BLNh[-1]
            else:
                BLNh_clean = BLNh[-2:]

            plt.figure()
            curr_title = "Blob = " + str(BLNh_clean) + ", rho = 1e-" + str(rhop[-1])
            curr_file = "nR_vs_Theta_Blob_" + str(BLNh_clean) + "_rho_" + str(rhop[-3:]) + ".pdf"
            plt.title(curr_title)
            plt.ylabel("Theta")
            plt.xlabel("nR")

            #Setting up colour cycle
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # for h, BLNh in enumerate(New[0][1][l][1]):
            for l, BFl in enumerate(New[0][1]):
                for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
                    try:
                        print("\n\nl, BFl = " + str(l) + ", " + str(New[0][1][l][0]))
                        print("p, rhop = " + str(p) + ", " + str(rhop))
                        print("h, BLNh = " + str(h) + ", " + str(BLNh))
                        # print("countt, aspect = " + str(countt) + ", " + str(aspect))
                        to_plot_NP = New[0][1][l][1][h][1][p][2][countt]
                        to_plot_PA = New[1][1][l][1][h][1][p][2][countt]


                    except Exception:
                        print("Exception!!")
                        continue

                    if len(aspect) == 6:
                        SCL_clean = aspect[-1]
                    else:
                        SCL_clean = aspect[-2:]


                    label_1 = "NP, BF=-" + str(BFl[0][-1]) + ", SCL=" + str(SCL_clean)
                    label_2 = "PA, BF=-" + str(BFl[0][-1]) + ", SCL=" + str(SCL_clean)
                    # color_count = l*6 + countt
                    color_count = countt
                    print("label_1 = " + str(label_1))
                    print("label_2 = " + str(label_2))


                    plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[color_count])
                    plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[color_count])


            plt.xscale("log")
            plt.legend(loc="best")
            plt.savefig((curr_file))
            plt.close()

    #Plotting for couple couple SCL / fugacity
    for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
        for p, rhop in enumerate(Rho_label_list):

            if len(aspect) == 6:
                SCL_clean = aspect[-1]
            else:
                SCL_clean = aspect[-2:]

            plt.figure()
            curr_title = "SCL = " + str(SCL_clean) + ", rho = 1e-" + str(rhop[-1])
            curr_file = "nR_vs_Theta_SCL_" + str(SCL_clean) + "_rho_" + str(rhop[-3:]) + ".pdf"
            plt.title(curr_title)
            plt.ylabel("Theta")
            plt.xlabel("nR")

            #Setting up colour cycle
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # for h, BLNh in enumerate(New[0][1][l][1]):
            for l, BFl in enumerate(New[0][1]):
                for h, BLNh in enumerate(blobN_label_list):
                    try:
                        BLNh = New[0][1][l][1][h][0]
                        print("\n\nl, BFl = " + str(l) + ", " + str(New[0][1][l][0]))
                        print("p, rhop = " + str(p) + ", " + str(rhop))
                        print("h, BLNh = " + str(h) + ", " + str(BLNh))
                        # print("countt, aspect = " + str(countt) + ", " + str(aspect))
                        to_plot_NP = New[0][1][l][1][h][1][p][2][countt]
                        to_plot_PA = New[1][1][l][1][h][1][p][2][countt]


                    except Exception:
                        print("Exception!!")
                        continue

                    if len(BLNh) == 6:
                        BLNh_clean = BLNh[-1]
                    else:
                        BLNh_clean = BLNh[-2:]

                    label_1 = "NP, BF=-" + str(BFl[0][-1]) + ", Blob=" + str(BLNh_clean)
                    label_2 = "PA, BF=-" + str(BFl[0][-1]) + ", Blob=" + str(BLNh_clean)
                    # color_count = l*4 + countt
                    color_count = h
                    print("label_1 = " + str(label_1))
                    print("label_2 = " + str(label_2))


                    plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[color_count])
                    plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[color_count])


            plt.xscale("log")
            plt.legend(loc="best")
            plt.savefig((curr_file))
            plt.close()

    ##
    #Varying fugacity
    for h, BLNh in enumerate(blobN_label_list):
        try:
            BLNh = New[0][1][0][1][h][0]
        except Exception:
            continue

        for l, BFl in enumerate(BF_label_list):
            try:
                BFl = New[0][1][l][0]
            except Exception:
                continue

            for countt, aspect in enumerate(SCL_label_list): #SCL_LABELS
                try:
                    if len(BLNh) == 6:
                        BLNh_clean = BLNh[-1]
                    else:
                        BLNh_clean = BLNh[-2:]

                    if len(aspect) == 6:
                        SCL_clean = aspect[-1]
                    else:
                        SCL_clean = aspect[-2:]

                    plt.figure()
                    curr_title = "Blob " + str(BLNh_clean) + ", BF = " + str(BFl[-1]) + ", SCL = " + str(SCL_clean)
                    curr_file ="nR_vs_Theta_Blob_" + str(BLNh_clean) + "_BF_" + str(BFl[-1]) + "_SCL_" + str(SCL_clean) + ".pdf"

                    plt.title(curr_title)
                    plt.ylabel("Theta")
                    plt.xlabel("nR")

                except Exception:
                    continue
                #Setting up colour cycle
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

                for p, rhop in enumerate(Rho_label_list):
                    try:
                        to_plot_NP = New[0][1][l][1][h][1][p][2][countt]
                        to_plot_PA = New[1][1][l][1][h][1][p][2][countt]
                    except Exception:
                        continue

                    label_1 = "NP, rho=1e-" + str(rhop[-1])
                    label_2 = "PA, rho=1e-" + str(rhop[-1:])

                    plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[p])
                    plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[p])


                plt.xscale("log")
                plt.legend(loc="best")
                plt.savefig((curr_file))
                plt.close()


    os.chdir('../')

    Plot_selectivity_PaNoPa(New)

    # os.makedirs('Superselectivity_plots', exist_ok=True)
    # os.chdir('Superselectivity_plots')
    # #Selectivity plot function
    # os.chdir('../')

    # os.makedirs('Colloid_alignment_plots', exist_ok=True)
    # os.chdir('Colloid_alignment_plots')
    # #Colloids alignment plot function
    # os.chdir('../')

    #############################################################################################

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


elif (PlotOnly):
    Just_Plot()
elif (Te_NewRule):    #!!DEBUGGING!!
    Plot_for_New_Rule_Test()    #!!DEBUGGING!!

elif (T_Surf_Re):    #!!DEBUGGING!!
    Plot_for_Surf_ext_Test()    #!!DEBUGGING!!

else:
    Plot_for_all_ZZ()

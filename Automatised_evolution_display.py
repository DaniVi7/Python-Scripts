#!/usr/bin/python
############################################################################################################################################################
############################################################################################################################################################
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial.polynomial import Polynomial
from itertools import islice
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d.axes3d import Axes3D
import os

import math
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

if args.Bulk: #Parsing arguments
    BULK = True #string
else:
    BULK = False

if args.Frequency: #Parsing arguments
    FREQUENCY = args.Frequency #string
else:
    FREQUENCY = 10

if args.All_Blob_Numbers: #Parsing arguments
    SCAN_BN = True #string
else:
    SCAN_BN = False

if args.Skip_Fraction: #Parsing arguments
    SKF = args.Skip_Fraction #string
else:
    SKF = 0.5

if args.Patch_No_Patch: #Parsing arguments
    PaNPa = True #string
else:
    PaNPa = False

############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################

def Read_evolution3D(FREQ):
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
    with open("step_vs_Nrho.log") as NR:
        with open("evolution.log") as EV:

            emptyline = EV.readline() #empty line in evolution file
            emptyline = NR.readline() #Empty line in Nrho file

            #Initialising data arrays
            StepN_arr = []
            En_arr = []
            NColl_arr = []
            NRho_arr = []
            Pack_arr = []
            line_counter = 0

            while True: #Try execpt to catch end of file
                try:
                    thisline = EV.readline() #Reading lines from both files
                    NRline = NR.readline()
                    #print(thisline)
                    thisline = thisline.split() #Splitting lines
                    NRline = NRline.split()

                    StepN = float(thisline[1][:-1]) #Reading values from evolution file
                    En = float(thisline[3][:-1])
                    NColl = float(thisline[-1])

                    NRho_i = float(NRline[4]) #Reading values from Nrho file
                    Pack_N = float(NRline[7])

                except Exception: #End of file or incomplete line
                    #print("Reached end of file!")
                    break

                else:
                    if (line_counter%FREQ) == 0: #Only storing values from files with required frequecny
                        StepN_arr.append(StepN)
                        En_arr.append(En)
                        NColl_arr.append(NColl)

                        NRho_arr.append(NRho_i)
                        Pack_arr.append(Pack_N)

                    line_counter += 1


            #"End of while loop"
            #print("Created arrays of data with required frequency!\n")



    return StepN_arr, En_arr, NColl_arr, NRho_arr, Pack_arr

##############################################################################

def Plot_data_smooth3D(STEPN_X, EN_Y, NCOLL_Y,  NRHO_Y, PACK_Y, SM_RANGE):
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
    SM_NRHO_Y = []
    SM_PACK_Y = []

    for i in range(SM_RANGE, (len(STEPN_X) - SM_RANGE)):
        SM_EN_Y.append(np.average(EN_Y[i - SM_RANGE : i + SM_RANGE] ) )
        SM_NCOLL_Y.append(np.average(NCOLL_Y[i - SM_RANGE : i + SM_RANGE] ) )
        SM_NRHO_Y.append(np.average(NRHO_Y[i - SM_RANGE : i + SM_RANGE] ) )
        SM_PACK_Y.append(np.average(PACK_Y[i - SM_RANGE : i + SM_RANGE] ) )


    AVG_En = np.average(EN_Y)
    AVG_NColl = np.average(NCOLL_Y)
    AVG_NRho = np.average(NRHO_Y)
    AVG_Pack = np.average(PACK_Y)
    ############################################################################

    ############################################################################
    #ENERGY PLOTTING
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
    #Nc PLOTTING
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
    #NRHO PLOTTING
    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.set_title("Number Density")
    ax1.set_ylabel(r"$\rho_n$")
    ax1.set_xlabel("Number of steps")
    ax1.plot(STEPN_X, NRHO_Y, color='b')
    ax1.axhline(y = AVG_NRho, color = 'b', linestyle = '-')

    ax2 = fig.add_subplot(212)
    ax2.set_title("Number Density Local Average")
    ax2.set_ylabel(r"$\rho_n$")
    ax2.set_xlabel("Number of steps")
    ax2.plot(STEPN_X[SM_RANGE:(len(STEPN_X) - SM_RANGE)], SM_NRHO_Y, color='b')
    ax2.axhline(y = AVG_NRho, color = 'b', linestyle = '-')

    fig.tight_layout()
    plt.savefig(("NRho_vs_Nstep.pdf"))
    plt.close()

    ############################################################################
    #PACKING PLOTTING
    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.set_title("Packing Fraction")
    ax1.set_ylabel(r"$\eta$")
    ax1.set_xlabel("Number of steps")
    ax1.plot(STEPN_X, PACK_Y, color='k')
    ax1.axhline(y = AVG_Pack, color = 'k', linestyle = '-')

    ax2 = fig.add_subplot(212)
    ax2.set_title("Packing Fraction Local Average")
    ax2.set_ylabel(r"$\eta$")
    ax2.set_xlabel("Number of steps")
    ax2.plot(STEPN_X[SM_RANGE:(len(STEPN_X) - SM_RANGE)], SM_PACK_Y, color='k')
    ax2.axhline(y = AVG_Pack, color = 'k', linestyle = '-')

    fig.tight_layout()
    plt.savefig(("Pack_vs_Nstep.pdf"))
    plt.close()

    ############################################################################
    ############################################################################

    with open("Averages.data", "w+") as A:
        A.write("Average Energy = " + str(AVG_En))
        A.write("\nAverage Number of colloids = " + str(AVG_NColl))
        A.write("\nAverage NRho = " + str(AVG_NRho))
        A.write("\nAverage Pack = " + str(AVG_Pack) + "\n")
        A.write("Number of steps analysed = " + str(STEPN_X[-1]))


    return

##############################################################################

def Read_reports3D():
    """!!!!!! ONLY READING THETA FOR NOW !!!!!
    """

    with open("Averages.data") as AV:

        Energy_line = AV.readline()
        NColl_line = AV.readline()
        NRho_line = AV.readline().split()
        Pack_line = AV.readline().split()

        N_DENS = NRho_line[3]
        PACK = Pack_line[3]

        return N_DENS, PACK

##############################################################################

def Plot_for_all_ZZ3D(TITLE):

    name_of_dirs = os.listdir()
    ZZ_Pack_list = []
    ZZ_list = []
    Restarted = False

    for i in name_of_dirs:
        this_point = []
        try:
            aaa = float(i)
        except Exception:
            #print("Reached end of file!")
            continue


        this_point.append(float(i)) #ZZ value
        os.chdir(i) #entering new directory with data
        print("entering new directory: " + str(i))
        inside_dir = os.listdir()
        # print("Check 1")
        if "RE" in inside_dir:
            # print("Check 2")
            os.chdir("RE") #entering new directory with data
            Restarted = True
        ##
        try:
            AllSteps, AllEn, AllNColl, AllNRho, AllPack = Read_evolution3D(FREQUENCY)
        except Exception:
            #print("Reached end of file!")
            os.chdir("../") #returning to initial directory
            print("exiting directory.. A")
            if Restarted==True:
                os.chdir("../") #returning to initial directory
                print("exiting directory.. B")
            continue
        ##

        Plot_data_smooth3D(AllSteps, AllEn, AllNColl, AllNRho, AllPack, SMOOTHING)
        # print("Check 2.1")
        this_nrho, this_pack = Read_reports3D()
        # print("Check 2.2")
        this_point.append(float(this_nrho))
        this_point.append(float(this_pack))
        this_point = np.array(this_point) #Pack value
        ZZ_Pack_list.append(this_point)
        os.chdir("../") #returning to initial directory
        # print("Check 3")

        if Restarted==True:
            os.chdir("../") #returning to initial directory
        ##
    ##End of for loop over directories


    ZZ_Pack_list = np.array(ZZ_Pack_list)
    # print("ZZ_Pack_list: ")
    # print(ZZ_Pack_list)

    ind = np.argsort(ZZ_Pack_list[:,0])
    ZZ_Pack_list = ZZ_Pack_list[ind]

    #print("ZZ_Pack_list: ")
    #print(ZZ_Pack_list)

    #fitted = np.polynomial.polynomial.Polynomial.fit(ZZ_Pack_list[:,1], ZZ_Pack_list[:,0], deg=4)
    #plotted = np.polynomial.polynomial.Polynomial()
    #ZZ_2perc = np.polyval(fitted, [0.02])
    #ZZ_02perc = np.polyval(fitted, [0.002])
    #ZZ_002perc = np.polyval(fitted, [0.0002])

    with open("ZZ_vs_Pack_recap.data", "w+") as TNR:
        #for counti, thetaj in enumerate(Theta_list):
        #    TNR.write("nR = " + str(Density_list[counti]) + ", Theta = "+ str(thetaj) + "\n")

        TNR.write("ZZ    NRho    Pack \n")

        for counti, element in enumerate(ZZ_Pack_list):
            #TNR.write(str(Dens_Theta_list[counti]) + "    "+ str(thetaj) + "\n")
            TNR.write(str(element[0]) + "    "+ str(element[1]) + "    "+ str(element[2]) + "\n")

        TNR.close()

    plt.figure()
    plt.ylabel(r"$\rho_n$")
    plt.xlabel("ZZ")
    plt.plot(ZZ_Pack_list[:,0], ZZ_Pack_list[:,1], color='g', linestyle="-", marker="o")
    PDFNAME = TITLE + "_NRho_vs_ZZ.pdf"
    plt.savefig(PDFNAME)
    plt.show()
    plt.close()


    plt.figure()
    plt.ylabel(r"$\eta$")
    plt.xlabel("ZZ")
    plt.plot(ZZ_Pack_list[:,0], ZZ_Pack_list[:,2], color='r', linestyle="-", marker="o")
    PDFNAME = TITLE + "_Pack_vs_ZZ.pdf"
    plt.savefig(PDFNAME)
    plt.show()
    plt.close()

    return ZZ_Pack_list

##############################################################################

def Plot_for_all_SC_L3D(PTCH, BLOB_N):

    dir_names = os.listdir()

    title_string = PTCH + ", NLig = " + BLOB_N[-1]
    file_string = PTCH + "_NLig_" + BLOB_N[-1]

    ZZ_vs_PACK_arr = []
    label_list = ["SC_L_6", "SC_L_3", "SC_L_0", "SC_L_9", "SC_L_12", "SC_L_15"]
    new_label_list = []

    for j in label_list:
    # for j in dir_names:
        #if (dir_names[:4] == "SC_L"):
        if j not in dir_names:
            continue

        title_string_2 = title_string + ", SC_L = " + j[-1]

        os.chdir(j)
        print("entering new directory: " + str(j))

        ##################################
        try:
            to_app = Plot_for_all_ZZ3D(title_string_2)
        except Exception:
            os.chdir("../")
            print("exiting directory.. E")
            continue
        ##################################

        ZZ_vs_PACK_arr.append(np.array(to_app))
        new_label_list.append(j)

        os.chdir("../")
        print("exiting directory.. G")

        continue
    #End of SC_L loop

    ZZ_vs_PACK_arr = np.array(ZZ_vs_PACK_arr)

    #Plotting for all SC_L
    plt.figure()
    plt.ylabel(r"$\rho_n$")
    plt.xlabel("ZZ")
    for k in range(len(new_label_list)):

        plt.plot(ZZ_vs_PACK_arr[k][:, 0], ZZ_vs_PACK_arr[k][:, 1], linestyle="-", marker="o", label=new_label_list[k])

    plt.legend(loc="best")
    PDF_NAME_1 = "SC_L_NRho_vs_ZZ_" + file_string + ".pdf"
    plt.savefig(PDF_NAME_1)

    plt.yscale("log")
    plt.xscale("log")
    PDF_NAME_2 = "LOG_SC_L_NRho_vs_ZZ_" + file_string + ".pdf"
    plt.savefig(PDF_NAME_2)
    plt.close()

    ##############
    plt.figure()
    plt.ylabel(r"$\eta$")
    plt.xlabel("ZZ")
    for k in range(len(new_label_list)):

        plt.plot(ZZ_vs_PACK_arr[k][:, 0], ZZ_vs_PACK_arr[k][:, 2], linestyle="-", marker="o", label=new_label_list[k])

    plt.legend(loc="best")
    PDF_NAME_1 = "SC_L_Pack_vs_ZZ" + file_string + ".pdf"
    plt.savefig(PDF_NAME_1)

    plt.yscale("log")
    plt.xscale("log")
    PDF_NAME_2 = "LOG_SC_L_Pack_vs_ZZ" + file_string + ".pdf"
    plt.savefig(PDF_NAME_2)
    plt.close()


    return new_label_list, ZZ_vs_PACK_arr

##############################################################################

def Plot_for_all_BN3D(Have_Patch):

    dir_names = os.listdir()
    list_of_results = []
    blobN_label = ["Blob_1", "Blob_3", "Blob_5", "Blob_10"]
    # blobN_label = ["Blob_1", "Blob_3", "Blob_5"]

    for j in blobN_label:
    # for j in dir_names:
        #if (dir_names[:4] == "SC_L"):
        if j not in dir_names:
            continue

        os.chdir(j)
        print("entering new directory: " + str(j))
        SCL_Labels, SCL_ZZ_vs_Pack = Plot_for_all_SC_L3D(Have_Patch, j)
        list_of_results.append(np.array([j, SCL_Labels, SCL_ZZ_vs_Pack]))
        os.chdir("../")
        continue
    ### End of loop over folders

    list_of_results = np.array(list_of_results)

    #PLOTTING DENSITY
    for countt, aspect in enumerate(list_of_results[0][1]):
        plt.figure()
        plt.title(aspect)
        plt.ylabel(r"$\rho_n$")
        plt.xlabel("ZZ")

        for k in range(len(blobN_label)):

            try:
                chh = list_of_results[k][2][countt]
            except Exception:
                continue

            plt.plot(list_of_results[k][2][countt][:, 0], list_of_results[k][2][countt][:, 1], linestyle="-", marker="o", label=blobN_label[k])

        plt.legend(loc="best")
        PDF_NAME_1 = "All_BN_" + aspect +  "_NRho_vs_ZZ_" + aspect + ".pdf"
        #plt.savefig(("All_SC_L_Theta_vs_nR.pdf"))
        plt.savefig((PDF_NAME_1))

        plt.yscale("log")
        plt.xscale("log")
        PDF_NAME_2 = "LOG_" + PDF_NAME_1
        plt.savefig(PDF_NAME_2)
        plt.close()

    #PLOTTING PACKING
    for countt, aspect in enumerate(list_of_results[0][1]):
        plt.figure()
        plt.title(aspect)
        plt.ylabel(r"$\eta$")
        plt.xlabel("ZZ")

        for k in range(len(blobN_label)):

            try:
                chh = list_of_results[k][2][countt]
            except Exception:
                continue

            plt.plot(list_of_results[k][2][countt][:, 0], list_of_results[k][2][countt][:, 2], linestyle="-", marker="o", label=blobN_label[k])

        plt.legend(loc="best")
        PDF_NAME_1 = "All_BN_" + aspect +  "_PACK_vs_ZZ_" + aspect + ".pdf"
        #plt.savefig(("All_SC_L_Theta_vs_nR.pdf"))
        plt.savefig((PDF_NAME_1))

        plt.yscale("log")
        plt.xscale("log")
        PDF_NAME_2 = "LOG_" + PDF_NAME_1
        plt.savefig(PDF_NAME_2)
        plt.close()

    return list_of_results

##############################################################################

def Plot_Patch_and_No_Patch3D():

    # dir_names = os.listdir()
    FINAL_RESULT_LIST = []

    #for j in ["Patch", "No_patch"]:
    for j in ["No_patch", "Patch"]:
    # for j in dir_names:
        #if (dir_names[:4] == "SC_L"):

        os.chdir(j)
        print("entering new directory: " + str(j))
        Configuration_results = Plot_for_all_BN3D(j)
        FINAL_RESULT_LIST.append(np.array([j, Configuration_results]))
        os.chdir("../")
        print("exiting directory.. H")

        continue

    FINAL_RESULT_LIST = np.array(FINAL_RESULT_LIST)

    #NOTE: From now on we assume that all combinations are present in the data..
    #if data is incomplete code below will give error..
    Bln_tot = len(FINAL_RESULT_LIST[0][1]) #Retrieving total number o ligand densities explored
    SCL_tot = len(FINAL_RESULT_LIST[0][1][0][1]) #Retrieving total number of aspect ratio considered

    #######################################################################
    #######################################################################
    #Plot comparison of Patch and No patch on same graph.. various combo

    #Patch continous vs No patch dashed, all Blob number for each SCL
    #PLOTTING DENSITY
    for scli in range(SCL_tot): #For each SC length
        SCL_label = FINAL_RESULT_LIST[0][1][0][1][scli]
        plt.figure()
        curr_title = "Patch vs No Patch, " + str(SCL_label)
        plt.title(curr_title)
        plt.ylabel(r"$\rho_n$")
        plt.xlabel("ZZ")

        #Setting up colour cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


        for bni in range(Bln_tot):
            BLN_label = FINAL_RESULT_LIST[0][1][bni][0]

            label_1 = "No patch, " + str(BLN_label)
            plt.plot(FINAL_RESULT_LIST[0][1][bni][2][scli][:, 0], FINAL_RESULT_LIST[0][1][bni][2][scli][:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[bni])

            label_2 = "Patch, " + str(BLN_label)
            plt.plot(FINAL_RESULT_LIST[1][1][bni][2][scli][:, 0], FINAL_RESULT_LIST[1][1][bni][2][scli][:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[bni])
            ##End of plotting loop

        plt.legend(loc="best")
        PDF_NAME_4 = "Patch_vs_No_patch_" + str(SCL_label) +  "_NRho_vs_ZZ_" + ".pdf"
        plt.savefig(PDF_NAME_4)

        plt.yscale("log")
        plt.xscale("log")
        PDF_NAME_4_2 = "LOG_" + PDF_NAME_4
        plt.savefig(PDF_NAME_4_2)
        plt.close()

    #PLOTTING PACKING
    for scli in range(SCL_tot): #For each SC length
        SCL_label = FINAL_RESULT_LIST[0][1][0][1][scli]
        plt.figure()
        curr_title = "Patch vs No Patch, " + str(SCL_label)
        plt.title(curr_title)
        plt.ylabel(r"$\eta$")
        plt.xlabel("ZZ")

        #Setting up colour cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


        for bni in range(Bln_tot):
            BLN_label = FINAL_RESULT_LIST[0][1][bni][0]

            to_plot_NP = FINAL_RESULT_LIST[0][1][bni][2][scli]
            to_plot_PA = FINAL_RESULT_LIST[1][1][bni][2][scli]

            ##Reducing dimensions
            to_plot_NP = to_plot_NP[to_plot_NP[:,2] < 0.03]
            to_plot_PA = to_plot_PA[to_plot_PA[:,2] < 0.03]

            label_1 = "No patch, " + str(BLN_label)
            plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 2], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[bni])

            label_2 = "Patch, " + str(BLN_label)
            plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 2], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[bni])
            ##End of plotting loop

        ### Adding horizontal lines for pack = 2%, 0.2%, 0,02%
        plt.axhline(y=0.02, color=color_cycle[SCL_tot], linestyle=":", label="Pack = 2%")
        plt.axhline(y=0.002, color=color_cycle[SCL_tot], linestyle=":", label="Pack = 0.2%")
        plt.axhline(y=0.0002, color=color_cycle[SCL_tot], linestyle=":", label="Pack = 0.02%")

        plt.legend(loc="best")
        PDF_NAME_4 = "Patch_vs_No_patch_" + str(SCL_label) +  "_PACK_vs_ZZ_" + ".pdf"
        plt.savefig(PDF_NAME_4)

        plt.yscale("log")
        plt.xscale("log")
        PDF_NAME_4_2 = "LOG_" + PDF_NAME_4
        plt.savefig(PDF_NAME_4_2)
        plt.close()

    #########################################################################

    #Patch continous vs No patch dashed, all SCL number for each Blob number
    #PLOTTING DENSITY
    for bni in range(Bln_tot):
        BLN_label = FINAL_RESULT_LIST[0][1][bni][0]
        plt.figure()
        curr_title = "Patch vs No Patch, " + str(BLN_label)
        plt.title(curr_title)
        plt.ylabel(r"$\rho_n$")
        plt.xlabel("ZZ")

        #Setting up colour cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for scli in range(SCL_tot): #For each SC length
            SCL_label = FINAL_RESULT_LIST[0][1][0][1][scli]

            label_1 = "No patch, " + str(SCL_label)
            plt.plot(FINAL_RESULT_LIST[0][1][bni][2][scli][:, 0], FINAL_RESULT_LIST[0][1][bni][2][scli][:, 1], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[scli])

            label_2 = "Patch, " + str(SCL_label)
            plt.plot(FINAL_RESULT_LIST[1][1][bni][2][scli][:, 0], FINAL_RESULT_LIST[1][1][bni][2][scli][:, 1], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[scli])
            ##End of plotting loop

        plt.legend(loc="best")
        PDF_NAME_5 = "Patch_vs_No_patch_" + str(BLN_label) +  "_NRho_vs_ZZ_" + ".pdf"
        plt.savefig(PDF_NAME_5)

        plt.yscale("log")
        plt.xscale("log")
        PDF_NAME_5_2 = "LOG_" + PDF_NAME_5
        plt.savefig(PDF_NAME_5_2)

        plt.close()

    #PLOTTING PACKING
    for bni in range(Bln_tot):
        BLN_label = FINAL_RESULT_LIST[0][1][bni][0]
        plt.figure()
        curr_title = "Patch vs No Patch, " + str(BLN_label)
        plt.title(curr_title)
        plt.ylabel(r"$\eta$")
        plt.xlabel("ZZ")

        #Setting up colour cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for scli in range(SCL_tot): #For each SC length
            SCL_label = FINAL_RESULT_LIST[0][1][0][1][scli]

            to_plot_NP = FINAL_RESULT_LIST[0][1][bni][2][scli]
            to_plot_PA = FINAL_RESULT_LIST[1][1][bni][2][scli]

            ##Reducing dimensions
            to_plot_NP = to_plot_NP[to_plot_NP[:,2] < 0.03]
            to_plot_PA = to_plot_PA[to_plot_PA[:,2] < 0.03]

            label_1 = "No patch, " + str(SCL_label)
            plt.plot(to_plot_NP[:, 0], to_plot_NP[:, 2], linestyle="--", marker="^", markerfacecolor="None", label=label_1, color=color_cycle[scli])

            label_2 = "Patch, " + str(SCL_label)
            plt.plot(to_plot_PA[:, 0], to_plot_PA[:, 2], linestyle="-", marker="o", markerfacecolor="None", label=label_2, color=color_cycle[scli])
            ##End of plotting loop

        ### Adding horizontal lines for pack = 2%, 0.2%, 0,02%
        plt.axhline(y=0.02, color=color_cycle[SCL_tot], linestyle=":", label="Pack = 2%")
        plt.axhline(y=0.002, color=color_cycle[SCL_tot], linestyle=":", label="Pack = 0.2%")
        plt.axhline(y=0.0002, color=color_cycle[SCL_tot], linestyle=":", label="Pack = 0.02%")

        ###
        plt.legend(loc="best")
        PDF_NAME_5 = "Patch_vs_No_patch_" + str(BLN_label) +  "_PACK_vs_ZZ_" + ".pdf"
        plt.savefig(PDF_NAME_5)

        plt.yscale("log")
        plt.xscale("log")
        PDF_NAME_5_2 = "LOG_" + PDF_NAME_5
        plt.savefig(PDF_NAME_5_2)

        plt.close()
    ##

    ## FITTING OF DATA ##
    # Final_data_frame = {}
    # Fitted_polynomials = []
    # Fug_Dens_target = []
    # Fug_Pack_target = []
    #

    Fit_final_arr(FINAL_RESULT_LIST)
    LOG_Fit_final_arr(FINAL_RESULT_LIST)



    return

##############################################################################

def Fit_final_arr(RESARRAY):

    Final_data_frame = {}
    Fitted_polynomials = []
    Fug_Dens_target = []
    Fug_Pack_target = []

    Bln_tot = len(RESARRAY[0][1]) #Retrieving total number o ligand densities explored
    SCL_tot = len(RESARRAY[0][1][0][1]) #Retrieving total number of aspect ratio considered

    for bni in range(Bln_tot):
        BLN_label = RESARRAY[0][1][bni][0]
        Blob_dict = {}

        for scli in range(SCL_tot): #For each SC length
            SCL_label = RESARRAY[0][1][0][1][scli]
            curr_label = BLN_label + ", " + SCL_label

            to_fit_NP = RESARRAY[0][1][bni][2][scli]
            to_fit_PA = RESARRAY[1][1][bni][2][scli]
            #Note: [:,2] -> Packing   //  [:,1] -> Density   //  [:,0] -> X

            #Applying condition on density
            to_fit_NP = to_fit_NP[to_fit_NP[:,1] < 5e-4]
            to_fit_PA = to_fit_PA[to_fit_PA[:,1] < 5e-4]

            #numpy.polynomial.polynomial.polyfit
            fitted_pol_NP_Dens, residui_1 = polyfit(to_fit_NP[:,1], to_fit_NP[:,0], deg=4, full=True)
            fitted_pol_NP_Pack, residui_2= polyfit(to_fit_NP[:,2], to_fit_NP[:,0], deg=4, full=True)

            fitted_pol_PA_Dens, residui_3 = polyfit(to_fit_PA[:,1], to_fit_PA[:,0], deg=4, full=True)
            fitted_pol_PA_Pack, residui_4 = polyfit(to_fit_PA[:,2], to_fit_PA[:,0], deg=4, full=True)

            residui_1 = residui_1[0] / (len(to_fit_NP)*np.average(to_fit_NP[:,0]))
            residui_2 = residui_2[0] / (len(to_fit_NP)*np.average(to_fit_NP[:,0]))
            residui_3 = residui_3[0] / (len(to_fit_PA)*np.average(to_fit_PA[:,0]))
            residui_4 = residui_4[0] / (len(to_fit_PA)*np.average(to_fit_PA[:,0]))

            print("Just fitted for: " + str(curr_label))
            print("No Patch Density summed squared error = " + str(residui_1))
            print("No Patch Pack summed squared error = " + str(residui_2))
            print("Patch Density summed squared error = " + str(residui_3))
            print("Patch Pack summed squared error = " + str(residui_4))

            ##
            ## Computing target values of density and packing
            ## No Patch
            ZZNP_rho_4 = Polynomial(fitted_pol_NP_Dens)(1e-4)
            ZZNP_rho_5 = Polynomial(fitted_pol_NP_Dens)(1e-5)
            ZZNP_rho_6 = Polynomial(fitted_pol_NP_Dens)(1e-6)

            ZZNP_pack_2 = Polynomial(fitted_pol_NP_Pack)(2e-3)

            ## Patch
            ZZPA_rho_4 = Polynomial(fitted_pol_PA_Dens)(1e-4)
            ZZPA_rho_5 = Polynomial(fitted_pol_PA_Dens)(1e-5)
            ZZPA_rho_6 = Polynomial(fitted_pol_PA_Dens)(1e-6)

            ZZPA_pack_2 = Polynomial(fitted_pol_PA_Pack)(2e-3)

            # NP_Results = {"ZZNP_rho_4" : ZZNP_rho_4, "ZZNP_rho_5" : ZZNP_rho_5, "ZZNP_rho_6" : ZZNP_rho_6, "ZZNP_pack_2" : ZZNP_pack_2}
            # PA_Results = {"ZZPA_rho_4" : ZZPA_rho_4, "ZZPA_rho_5" : ZZPA_rho_5, "ZZPA_rho_6" : ZZPA_rho_6, "ZZPA_pack_2" : ZZPA_pack_2}
            # NP_Results = [{"rho":1e-4, "ZZ":  ZZNP_rho_4}, {"rho":1e-5, "ZZ":  ZZNP_rho_5}, {"rho":1e-6, "ZZ":  ZZNP_rho_6}, {"Pack":2e-3, "ZZ":  ZZNP_pack_2}]
            NP_Results = {}
            NP_Results["rho_1e-4"] = ZZNP_rho_4
            NP_Results["rho_1e-5"] = ZZNP_rho_5
            NP_Results["rho_1e-6"] = ZZNP_rho_6
            NP_Results["Pack_2e-6"] = ZZNP_pack_2



            # PA_Results = {"ZZPA_rho_4" : ZZPA_rho_4, "ZZPA_rho_5" : ZZPA_rho_5, "ZZPA_rho_6" : ZZPA_rho_6, "ZZPA_pack_2" : ZZPA_pack_2}
            # PA_Results = [{"rho":1e-4, "ZZ":  ZZPA_rho_4}, {"rho":1e-5, "ZZ":  ZZPA_rho_5}, {"rho":1e-6, "ZZ":  ZZPA_rho_6}, {"Pack":2e-3, "ZZ":  ZZPA_pack_2}]
            PA_Results = {}
            PA_Results["rho_1e-4"] = ZZPA_rho_4
            PA_Results["rho_1e-5"] = ZZPA_rho_5
            PA_Results["rho_1e-6"] = ZZPA_rho_6
            PA_Results["Pack_2e-6"] = ZZPA_pack_2


            # ResUnion = {"No Patch" : NP_Results, "Patch" : PA_Results}
            # Final_data_frame[curr_label] = ResUnion
            lbl1 = "No Patch " + SCL_label
            lbl2 = "Patch " + SCL_label

            # Blob_dict[SCL_label] = ResUnion
            Blob_dict[lbl1] = NP_Results
            Blob_dict[lbl2] = PA_Results


        Final_data_frame[BLN_label] = Blob_dict


    FDF = pd.DataFrame.from_dict(Final_data_frame, orient="index")
    FDF.to_csv("Fitting results.csv")

    return

##############################################################################

def LOG_Fit_final_arr(RESARRAY):

    Final_data_frame = {}
    Fitted_polynomials = []
    Fug_Dens_target = []
    Fug_Pack_target = []

    Bln_tot = len(RESARRAY[0][1]) #Retrieving total number o ligand densities explored
    SCL_tot = len(RESARRAY[0][1][0][1]) #Retrieving total number of aspect ratio considered

    for bni in range(Bln_tot):
        BLN_label = RESARRAY[0][1][bni][0]
        Blob_dict = {}

        for scli in range(SCL_tot): #For each SC length
            SCL_label = RESARRAY[0][1][0][1][scli]
            curr_label = BLN_label + ", " + SCL_label

            to_fit_NP = RESARRAY[0][1][bni][2][scli]
            to_fit_PA = RESARRAY[1][1][bni][2][scli]
            #Note: [:,2] -> Packing   //  [:,1] -> Density   //  [:,0] -> X

            #Applying condition on density
            try:
                to_fit_NP = np.log(to_fit_NP)
                to_fit_PA = np.log(to_fit_PA)

                #numpy.polynomial.polynomial.polyfit
                fitted_pol_NP_Dens, residui_1 = polyfit(to_fit_NP[:,1], to_fit_NP[:,0], deg=2, full=True)
                fitted_pol_NP_Pack, residui_2= polyfit(to_fit_NP[:,2], to_fit_NP[:,0], deg=2, full=True)

                fitted_pol_PA_Dens, residui_3 = polyfit(to_fit_PA[:,1], to_fit_PA[:,0], deg=2, full=True)
                fitted_pol_PA_Pack, residui_4 = polyfit(to_fit_PA[:,2], to_fit_PA[:,0], deg=2, full=True)

                residui_1 = residui_1[0] / (len(to_fit_NP)*np.average(to_fit_NP[:,0]))
                residui_2 = residui_2[0] / (len(to_fit_NP)*np.average(to_fit_NP[:,0]))
                residui_3 = residui_3[0] / (len(to_fit_PA)*np.average(to_fit_PA[:,0]))
                residui_4 = residui_4[0] / (len(to_fit_PA)*np.average(to_fit_PA[:,0]))

            except Exception: #End of file or incomplete line
                print("Error at " + curr_label)
                continue

            print("Just fitted for: " + str(curr_label))
            print("No Patch Density summed squared error = " + str(residui_1))
            print("No Patch Pack summed squared error = " + str(residui_2))
            print("Patch Density summed squared error = " + str(residui_3))
            print("Patch Pack summed squared error = " + str(residui_4))

            ##
            ## Computing target values of density and packing
            ## No Patch
            ZZNP_rho_4 = np.exp(Polynomial(fitted_pol_NP_Dens)(np.log(1e-4)))
            ZZNP_rho_5 = np.exp(Polynomial(fitted_pol_NP_Dens)(np.log(1e-5)))
            ZZNP_rho_6 = np.exp(Polynomial(fitted_pol_NP_Dens)(np.log(1e-6)))

            ZZNP_pack_2 = np.exp(Polynomial(fitted_pol_NP_Pack)(np.log(2e-3)))

            ## Patch
            ZZPA_rho_4 = np.exp(Polynomial(fitted_pol_PA_Dens)(np.log(1e-4)))
            ZZPA_rho_5 = np.exp(Polynomial(fitted_pol_PA_Dens)(np.log(1e-5)))
            ZZPA_rho_6 = np.exp(Polynomial(fitted_pol_PA_Dens)(np.log(1e-6)))

            ZZPA_pack_2 = np.exp(Polynomial(fitted_pol_PA_Pack)(np.log(2e-3)))

            # NP_Results = {"ZZNP_rho_4" : ZZNP_rho_4, "ZZNP_rho_5" : ZZNP_rho_5, "ZZNP_rho_6" : ZZNP_rho_6, "ZZNP_pack_2" : ZZNP_pack_2}
            # NP_Results = [{"rho":1e-4, "ZZ":  ZZNP_rho_4}, {"rho":1e-5, "ZZ":  ZZNP_rho_5}, {"rho":1e-6, "ZZ":  ZZNP_rho_6}, {"Pack":2e-3, "ZZ":  ZZNP_pack_2}]
            # NP_Results = [{"rho":1e-4, "ZZ":  ZZNP_rho_4}, {"rho":1e-5, "ZZ":  ZZNP_rho_5}, {"rho":1e-6, "ZZ":  ZZNP_rho_6}, {"Pack":2e-3, "ZZ":  ZZNP_pack_2}]
            NP_Results = {}
            NP_Results["rho_1e-4"] = ZZNP_rho_4
            NP_Results["rho_1e-5"] = ZZNP_rho_5
            NP_Results["rho_1e-6"] = ZZNP_rho_6
            NP_Results["Pack_2e-6"] = ZZNP_pack_2

            # PA_Results = {"ZZPA_rho_4" : ZZPA_rho_4, "ZZPA_rho_5" : ZZPA_rho_5, "ZZPA_rho_6" : ZZPA_rho_6, "ZZPA_pack_2" : ZZPA_pack_2}
            # PA_Results = [{"rho":1e-4, "ZZ":  ZZPA_rho_4}, {"rho":1e-5, "ZZ":  ZZPA_rho_5}, {"rho":1e-6, "ZZ":  ZZPA_rho_6}, {"Pack":2e-3, "ZZ":  ZZPA_pack_2}]
            PA_Results = {}
            PA_Results["rho_1e-4"] = ZZPA_rho_4
            PA_Results["rho_1e-5"] = ZZPA_rho_5
            PA_Results["rho_1e-6"] = ZZPA_rho_6
            PA_Results["Pack_2e-6"] = ZZPA_pack_2



            # ResUnion = {"No Patch" : NP_Results, "Patch" : PA_Results}
            # Final_data_frame[curr_label] = ResUnion
            lbl1 = "No Patch " + SCL_label
            lbl2 = "Patch " + SCL_label

            # Blob_dict[SCL_label] = ResUnion
            Blob_dict[lbl1] = NP_Results
            Blob_dict[lbl2] = PA_Results


        Final_data_frame[BLN_label] = Blob_dict


    FDF = pd.DataFrame.from_dict(Final_data_frame, orient="index")
    FDF.to_csv("Fitting LOG results.csv")

    return

##############################################################################
##############################################################################
##############################################################################

## MAIN ##

#AllSteps, AllEn, AllNColl, AllPack = Read_evolution3D(FREQUENCY)
#Plot_data_smooth3D(AllSteps, AllEn, AllNColl, AllPack, SMOOTHING)
if (SCL == True):
    Plot_for_all_SC_L3D()

elif (SCAN_BN):
    Plot_for_all_BN3D()

elif (PaNPa):
    Plot_Patch_and_No_Patch3D()

else:
    Plot_for_all_ZZ3D()






    # for bni in range(Bln_tot):
    #     BLN_label = FINAL_RESULT_LIST[0][1][bni][0]
    #
    #     for scli in range(SCL_tot): #For each SC length
    #         SCL_label = FINAL_RESULT_LIST[0][1][0][1][scli]
    #         curr_label = BLN_label + ", " + SCL_label
    #
    #         to_fit_NP = FINAL_RESULT_LIST[0][1][bni][2][scli]
    #         to_fit_PA = FINAL_RESULT_LIST[1][1][bni][2][scli]
    #         #Note: [:,2] -> Packing   //  [:,1] -> Density   //  [:,0] -> X
    #
    #         #Applying condition on density
    #         to_fit_NP = to_fit_NP[to_fit_NP[:,1] < 5e-4]
    #         to_fit_PA = to_fit_PA[to_fit_PA[:,1] < 5e-4]
    #
    #         #numpy.polynomial.polynomial.polyfit
    #         fitted_pol_NP_Dens, residui_1 = polyfit(to_fit_NP[:,1], to_fit_NP[:,0], deg=4, full=True)
    #         fitted_pol_NP_Pack, residui_2= polyfit(to_fit_NP[:,2], to_fit_NP[:,0], deg=4, full=True)
    #
    #         fitted_pol_PA_Dens, residui_3 = polyfit(to_fit_PA[:,1], to_fit_PA[:,0], deg=4, full=True)
    #         fitted_pol_PA_Pack, residui_4 = polyfit(to_fit_PA[:,2], to_fit_PA[:,0], deg=4, full=True)
    #
    #         residui_1 = residui_1[0] / (len(to_fit_NP)*np.average(to_fit_NP[:,0]))
    #         residui_2 = residui_2[0] / (len(to_fit_NP)*np.average(to_fit_NP[:,0]))
    #         residui_3 = residui_3[0] / (len(to_fit_PA)*np.average(to_fit_PA[:,0]))
    #         residui_4 = residui_4[0] / (len(to_fit_PA)*np.average(to_fit_PA[:,0]))
    #
    #         print("Just fitted for: " + str(curr_label))
    #         print("No Patch Density summed squared error = " + str(residui_1))
    #         print("No Patch Pack summed squared error = " + str(residui_2))
    #         print("Patch Density summed squared error = " + str(residui_3))
    #         print("Patch Pack summed squared error = " + str(residui_4))
    #
    #         ##
    #         ## Computing target values of density and packing
    #         ## No Patch
    #         ZZNP_rho_4 = Polynomial(fitted_pol_NP_Dens)(1e-4)
    #         ZZNP_rho_5 = Polynomial(fitted_pol_NP_Dens)(1e-5)
    #         ZZNP_rho_6 = Polynomial(fitted_pol_NP_Dens)(1e-6)
    #
    #         ZZNP_pack_2 = Polynomial(fitted_pol_NP_Pack)(2e-3)
    #
    #         ## Patch
    #         ZZPA_rho_4 = Polynomial(fitted_pol_PA_Dens)(1e-4)
    #         ZZPA_rho_5 = Polynomial(fitted_pol_PA_Dens)(1e-5)
    #         ZZPA_rho_6 = Polynomial(fitted_pol_PA_Dens)(1e-6)
    #
    #         ZZPA_pack_2 = Polynomial(fitted_pol_PA_Pack)(2e-3)
    #
    #         NP_Results = {"ZZNP_rho_4" : ZZNP_rho_4, "ZZNP_rho_5" : ZZNP_rho_5, "ZZNP_rho_6" : ZZNP_rho_6, "ZZNP_pack_2" : ZZNP_pack_2}
    #         PA_Results = {"ZZPA_rho_4" : ZZPA_rho_4, "ZZPA_rho_5" : ZZPA_rho_5, "ZZPA_rho_6" : ZZPA_rho_6, "ZZPA_pack_2" : ZZPA_pack_2}
    #         ResUnion = {"No Patch" : NP_Results, "Patch" : PA_Results}
    #         Final_data_frame[curr_label] = ResUnion
    #
    #
    # FDF = pd.DataFrame.from_dict(Final_data_frame, orient="index")
    # FDF.to_csv("Fitting results.csv")
    #

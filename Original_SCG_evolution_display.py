#!/usr/bin/python

import numpy as np
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



parser = argparse.ArgumentParser()
parser.add_argument("-F", "--File_name", help="The name of the evolution file from which to extract the data", type=str)
parser.add_argument("-fr", "--Frequency", help="The period with which data point are extracted from file (inverse of frequency)", type=float)
parser.add_argument("-SM", "--Smoothing", help="The number of point considered before and after each data point for smoothing", type=int)
parser.add_argument("-NE", "--no_energy", help="The number of point considered before and after each data point for smoothing", type=bool)
parser.add_argument('-FP', '--final_plot', help="Final plot of extracted average thetas on a single graph", type=bool)

args = parser.parse_args()


if args.File_name: #Parsing arguments
    Evolution_file = args.File_name #string
else:
    Evolution_file = "evolution.log"

if args.Frequency: #Parsing arguments
    FREQUENCY = args.Frequency #string
else:
    FREQUENCY = 1

if args.Frequency: #Parsing arguments
    SMOOTHING = args.Smoothing #string
else:
    SMOOTHING = 50

if args.no_energy: #Parsing arguments
    NO_ENERGY = args.no_energy #string
else:
    NO_ENERGY = False

if args.final_plot: #Parsing arguments
    FINALP = True #string
else:
    FINALP = False


############################################################################################################################################################
############################################################################################################################################################


def Read_evolution(FREQ, FNAME):
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

    with open(FNAME) as EV:

        emptyline = EV.readline()
        StepN_arr = []
        En_arr = []
        #NPart_arr = []
        NBonds_arr = []
        NColl_arr = []
        Theta_arr = []
        line_counter = 0

        while True:
            try:
                thisline = EV.readline()
                #print(thisline)
                thisline = thisline.split()
                StepN = float(thisline[1][:-1])

            except:
                print("Reached end of file!")
                break

            else:
                if (line_counter%FREQ) == 0:
                    StepN_arr.append(StepN)
                    En = float(thisline[3][:-1])
                    En_arr.append(En)
                    #NPart =
                    NBonds = float(thisline[9][:-1])
                    NBonds_arr.append(NBonds)
                    NColl = float(thisline[15][:-1])
                    NColl_arr.append(NColl)
                    Theta = float(thisline[18])
                    Theta_arr.append(Theta)

                emptyline = EV.readline()
                emptyline = EV.readline()

                line_counter += 1


        "End of while loop"
        print("Created arrays of data with required frequency!\n")



    return StepN_arr, En_arr, NBonds_arr, NColl_arr, Theta_arr


##############################################################################

def Plot_data(STEPN_X, EN_Y, NBONDS_Y, NCOLL_Y, THETA_Y):
    """Function reading standard evolution file from Sferocylinder simulations and extracting data. Returns arrays with data extracted
    with required Frequency
    Inputs: - STEPN_X, float array: the array of the step number of all data points sampled;
            - EN_Y, float array: the array of the energy of all data points sampled;
            - NBONDS_Y, float array: the array of the number of bonds of all data points sampled;
            - NCOLL_Y, float array: the array of the number of adsorbed colloids of all data points sampled;
            - THETA_Y, float array: the array of Theta of all data points sampled.
    Returns: None.
    """

    print("Plotting the Energy evolution.. ")
    plt.figure()
    plt.ylabel("Total Energy")
    plt.xlabel("Number of steps")
    plt.plot(STEPN_X, EN_Y, color='r')
    plt.savefig(("TotE_vs_Nstep.pdf"))
    plt.show()
    plt.close()

    print("Plotting the Number of Bonds evolution.. ")
    plt.figure()
    plt.ylabel("Total Energy")
    plt.xlabel("Number of bonds")
    plt.plot(STEPN_X, NBONDS_Y, color='b')
    plt.savefig(("NBonds_vs_Nstep.pdf"))
    plt.show()
    plt.close()

    print("Plotting the Number of Colloids evolution.. ")
    plt.figure()
    plt.ylabel("Number of Ads. Colloids")
    plt.xlabel("Number of steps")
    plt.plot(STEPN_X, NCOLL_Y, color='g')
    plt.savefig(("NColl_vs_Nstep.pdf"))
    plt.show()
    plt.close()


    print("Plotting the Theta evolution.. ")
    plt.figure()
    plt.ylabel("Theta")
    plt.xlabel("Number of steps")
    plt.plot(STEPN_X, THETA_Y, color='k')
    plt.savefig(("Theta_vs_Nstep.pdf"))
    plt.show()
    plt.close()

    return

##############################################################################

def Plot_data_smooth(STEPN_X, EN_Y, NBONDS_Y, NCOLL_Y, THETA_Y, SM_RANGE):
    """Function reading standard evolution file from Sferocylinder simulations and extracting data. Returns arrays with data extracted
    with required Frequency
    Inputs: - STEPN_X, float array: the array of the step number of all data points sampled;
            - EN_Y, float array: the array of the energy of all data points sampled;
            - NBONDS_Y, float array: the array of the number of bonds of all data points sampled;
            - NCOLL_Y, float array: the array of the number of adsorbed colloids of all data points sampled;
            - THETA_Y, float array: the array of Theta of all data points sampled.
    Returns: None.
    """
    SM_EN_Y = []
    SM_NBONDS_Y = []
    SM_NCOLL_Y = []
    SM_THETA_Y = []

    for i in range(SM_RANGE, (len(STEPN_X) - SM_RANGE)):
        SM_EN_Y.append(np.average(EN_Y[i - SM_RANGE : i + SM_RANGE] ) )
        SM_NBONDS_Y.append(np.average(NBONDS_Y[i - SM_RANGE : i + SM_RANGE] ) )
        SM_NCOLL_Y.append(np.average(NCOLL_Y[i - SM_RANGE : i + SM_RANGE] ) )
        SM_THETA_Y.append(np.average(THETA_Y[i - SM_RANGE : i + SM_RANGE] ) )


    AVG_En = np.average(EN_Y)
    print("Plotting the Energy evolution.. ")
    plt.figure()
    plt.ylabel("Total Energy")
    plt.xlabel("Number of steps")
    plt.plot(STEPN_X[SM_RANGE:(len(STEPN_X) - SM_RANGE)], SM_EN_Y, color='r')
    plt.axhline(y = AVG_En, color = 'r', linestyle = '-')
    plt.savefig(("TotE_vs_Nstep.pdf"))
    plt.show()
    plt.close()

    AVG_NBonds = np.average(NBONDS_Y)
    print("Plotting the Number of Bonds evolution.. ")
    plt.figure()
    plt.ylabel("Total Energy")
    plt.xlabel("Number of bonds")
    plt.plot(STEPN_X[SM_RANGE:(len(STEPN_X) - SM_RANGE)], SM_NBONDS_Y, color='b')
    plt.axhline(y = AVG_NBonds, color = 'b', linestyle = '-')
    plt.savefig(("NBonds_vs_Nstep.pdf"))
    plt.show()
    plt.close()

    AVG_NColl = np.average(NCOLL_Y)
    print("Plotting the Number of Colloids evolution.. ")
    plt.figure()
    plt.ylabel("Number of Ads. Colloids")
    plt.xlabel("Number of steps")
    plt.plot(STEPN_X[SM_RANGE:(len(STEPN_X) - SM_RANGE)], SM_NCOLL_Y, color='g')
    plt.axhline(y = AVG_NColl, color = 'g', linestyle = '-')
    plt.savefig(("NColl_vs_Nstep.pdf"))
    plt.show()
    plt.close()

    AVG_Theta = np.average(THETA_Y)
    print("Plotting the Theta evolution.. ")
    plt.figure()
    plt.ylabel("Theta")
    plt.xlabel("Number of steps")
    plt.plot(STEPN_X[SM_RANGE:(len(STEPN_X) - SM_RANGE)], SM_THETA_Y, color='k')
    plt.axhline(y = AVG_Theta, color = 'k', linestyle = '-')
    plt.savefig(("Theta_vs_Nstep.pdf"))
    plt.show()
    plt.close()

    with open("Averages.data", "w+") as A:
        A.write("Average Energy = " + str(AVG_En))
        A.write("\nAverage Number of bonds = " + str(AVG_NBonds))
        A.write("\nAverage Number of Adsorbed colloids = " + str(AVG_NColl))
        A.write("\nAverage Theta = " + str(AVG_Theta) + "\n")


    return


##############################################################################

def Plot_data_smooth_noEn(STEPN_X, EN_Y, NBONDS_Y, NCOLL_Y, THETA_Y, SM_RANGE):
    """Function reading standard evolution file from Sferocylinder simulations and extracting data. Returns arrays with data extracted
    with required Frequency
    Inputs: - STEPN_X, float array: the array of the step number of all data points sampled;
            - EN_Y, float array: the array of the energy of all data points sampled;
            - NBONDS_Y, float array: the array of the number of bonds of all data points sampled;
            - NCOLL_Y, float array: the array of the number of adsorbed colloids of all data points sampled;
            - THETA_Y, float array: the array of Theta of all data points sampled.
    Returns: None.
    """
    #SM_EN_Y = []
    SM_NBONDS_Y = []
    SM_NCOLL_Y = []
    SM_THETA_Y = []

    for i in range(SM_RANGE, (len(STEPN_X) - SM_RANGE)):
        #SM_EN_Y.append(np.average(EN_Y[i - SM_RANGE : i + SM_RANGE] ) )
        SM_NBONDS_Y.append(np.average(NBONDS_Y[i - SM_RANGE : i + SM_RANGE] ) )
        SM_NCOLL_Y.append(np.average(NCOLL_Y[i - SM_RANGE : i + SM_RANGE] ) )
        SM_THETA_Y.append(np.average(THETA_Y[i - SM_RANGE : i + SM_RANGE] ) )


    #AVG_En = np.average(EN_Y)
    #print("Plotting the Energy evolution.. ")
    #plt.figure()
    #plt.ylabel("Total Energy")
    #plt.xlabel("Number of steps")
    #plt.plot(STEPN_X[SM_RANGE:(len(STEPN_X) - SM_RANGE)], SM_EN_Y, color='r')
    #plt.axhline(y = AVG_En, color = 'r', linestyle = '-')
    #plt.savefig(("TotE_vs_Nstep.pdf"))
    #plt.show()
    #plt.close()

    AVG_NBonds = np.average(NBONDS_Y)
    print("Plotting the Number of Bonds evolution.. ")
    plt.figure()
    plt.ylabel("Total Energy")
    plt.xlabel("Number of bonds")
    plt.plot(STEPN_X[SM_RANGE:(len(STEPN_X) - SM_RANGE)], SM_NBONDS_Y, color='b')
    plt.axhline(y = AVG_NBonds, color = 'b', linestyle = '-')
    plt.savefig(("NBonds_vs_Nstep.pdf"))
    plt.show()
    plt.close()

    AVG_NColl = np.average(NCOLL_Y)
    print("Plotting the Number of Colloids evolution.. ")
    plt.figure()
    plt.ylabel("Number of Ads. Colloids")
    plt.xlabel("Number of steps")
    plt.plot(STEPN_X[SM_RANGE:(len(STEPN_X) - SM_RANGE)], SM_NCOLL_Y, color='g')
    plt.axhline(y = AVG_NColl, color = 'g', linestyle = '-')
    plt.savefig(("NColl_vs_Nstep.pdf"))
    plt.show()
    plt.close()

    AVG_Theta = np.average(THETA_Y)
    print("Plotting the Theta evolution.. ")
    plt.figure()
    plt.ylabel("Theta")
    plt.xlabel("Number of steps")
    plt.plot(STEPN_X[SM_RANGE:(len(STEPN_X) - SM_RANGE)], SM_THETA_Y, color='k')
    plt.axhline(y = AVG_Theta, color = 'k', linestyle = '-')
    plt.savefig(("Theta_vs_Nstep.pdf"))
    plt.show()
    plt.close()

    with open("Averages.data", "w+") as A:
        #A.write("Average Energy = " + str(AVG_En))
        A.write("\nAverage Number of bonds = " + str(AVG_NBonds))
        A.write("\nAverage Number of Adsorbed colloids = " + str(AVG_NColl))
        A.write("\nAverage Theta = " + str(AVG_Theta) + "\n")


    return

##############################################################################

def Read_reports():
    """!!!!!! ONLY READING THETA FOR NOW !!!!!
    """

    with open("Averages.data") as AV:

        Energy_line = AV.readline()

        Bonds_line = AV.readline()

        NColl_line = AV.readline()

        Theta_line = AV.readline().split()
        THETA = Theta_line[3]

        return THETA


##############################################################################

def Plot_final_graph():

    name_of_dirs = os.listdir()
    Theta_list = []
    Density_list = []
    Restarted = False

    for i in name_of_dirs:
        if (i[:3] == "nR_"):
            if (i[3:] == "05" ):
                Density_list.append(0.5)
            else:
                Density_list.append(float(i[3:]))
            #

            os.chdir(i) #entering new directory with data

            inside_dir = os.listdir()
            if "RE" in inside_dir:
                os.chdir("RE") #entering new directory with data
                Restarted = True

            this_th = Read_reports()
            Theta_list.append(float(this_th))
            os.chdir("../") #returning to initial directory
            if Restarted==True:
                os.chdir("../") #returning to initial directory


    save_ds = 0
    for ds in range(len(Density_list)):
        if Density_list[ds] == 10:
            save_ds = ds
            break

    dieci = Density_list.pop(ds)
    Density_list.append(dieci)

    Theta_dieci = Theta_list.pop(ds)
    Theta_list.append(Theta_dieci)

    with open("Theta_vs_nR_recap.data", "w+") as TNR:
        #for counti, thetaj in enumerate(Theta_list):
        #    TNR.write("nR = " + str(Density_list[counti]) + ", Theta = "+ str(thetaj) + "\n")

        TNR.write("nR    Theta \n")

        for counti, thetaj in enumerate(Theta_list):
            TNR.write(str(Density_list[counti]) + "    "+ str(thetaj) + "\n")

        TNR.close()

    plt.figure()
    plt.ylabel("Theta")
    plt.xlabel("nR")
    plt.plot(Density_list, Theta_list, color='r', linestyle="-", marker="o")
    plt.savefig(("Theta_vs_nR.pdf"))
    plt.show()
    plt.close()

    return

##############################################################################

def Read_Average_Plot_final_graph():

    name_of_dirs = os.listdir()
    Theta_list = []
    Density_list = []
    Restarted = False

    for i in name_of_dirs:
        if (i[:3] == "nR_"):
            if (i[3:] == "05" ):
                Density_list.append(0.5)
            else:
                Density_list.append(float(i[3:]))
            #

            os.chdir(i) #entering new directory with data

            inside_dir = os.listdir()
            if "RE" in inside_dir:
                os.chdir("RE") #entering new directory with data
                Restarted = True


            AllSteps, AllEn, AllNBonds, AllNColl, AllTheta = Read_evolution(FREQUENCY, Evolution_file)
            Plot_data_smooth(AllSteps, AllEn, AllNBonds, AllNColl, AllTheta, SMOOTHING)
            this_th = Read_reports()
            Theta_list.append(float(this_th))
            os.chdir("../") #returning to initial directory
            if Restarted==True:
                os.chdir("../") #returning to initial directory


    save_ds = 0
    for ds in range(len(Density_list)):
        if Density_list[ds] == 10:
            save_ds = ds
            break

    dieci = Density_list.pop(ds)
    Density_list.append(dieci)

    Theta_dieci = Theta_list.pop(ds)
    Theta_list.append(Theta_dieci)

    with open("Theta_vs_nR_recap.data", "w+") as TNR:
        #for counti, thetaj in enumerate(Theta_list):
        #    TNR.write("nR = " + str(Density_list[counti]) + ", Theta = "+ str(thetaj) + "\n")

        TNR.write("nR    Theta \n")

        for counti, thetaj in enumerate(Theta_list):
            TNR.write(str(Density_list[counti]) + "    "+ str(thetaj) + "\n")

        TNR.close()

    plt.figure()
    plt.ylabel("Theta")
    plt.xlabel("nR")
    plt.plot(Density_list, Theta_list, color='r', linestyle="-", marker="o")
    plt.savefig(("Theta_vs_nR.pdf"))
    plt.show()
    plt.close()

    return

##############################################################################

##############################################################################
######################           MAIN            #############################
##############################################################################

if (FINALP == False):
    AllSteps, AllEn, AllNBonds, AllNColl, AllTheta = Read_evolution(FREQUENCY, Evolution_file)
    #Plot_data(AllSteps, AllEn, AllNBonds, AllNColl, AllTheta)
    if (NO_ENERGY == True):
        Plot_data_smooth_noEn(AllSteps, AllEn, AllNBonds, AllNColl, AllTheta, SMOOTHING)
    else:
        Plot_data_smooth(AllSteps, AllEn, AllNBonds, AllNColl, AllTheta, SMOOTHING)

else:
    print("Plotting final graph")
    Plot_final_graph()




print("\n!!  THE END  !!\n")

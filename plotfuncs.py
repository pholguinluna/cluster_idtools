""" This contains all of the functions for plotting """

from pyigm.guis import igmguesses
from linetools.lists.linelist import LineList
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from archivalCGMtools import utils as au
from importlib import reload
from astropy.io import ascii
from astropy import units as u
from linetools.spectra.io import readspec
from linetools.spectralline import AbsLine
from linetools.isgm.abscomponent import AbsComponent
from collections import OrderedDict
from linetools import utils as lu



def plotEW(galdict, maintab, colors, EW='summed'):
    """
    This function takes a dictionary whose indices correspond to those of the maintable used to create the dict, the
    maintable, and an array of different colors the same length as the maintable, to create a plot of the 
    EW vs. RRVIR. The error in the EW measurement is accounted for as an errorbar on the plot, and if the detection is
    statistically significant (flag = 1) it is plotted as a circle, but if it is a non-detection (flag = 3), it is 
    plotted as a downward arrow.
    
    """
    # csfont = {'fontname':'Times New Roman'}
    
    plt.figure(figsize=(10,7),)
    # plt.xlabel('r/r$_{vir}$', fontsize=30, **csfont)
    # plt.ylabel('HI EW ($\AA$)', fontsize=30, **csfont)
    # plt.yscale('log')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
#     plt.title(r'Lyman $\alpha$ as a function of rrvir', fontsize=20)

    if EW == 'individual':
        for i,si in enumerate(galdict['sysid']):
            ews = galdict['EW'][i]
            sigews = galdict['sig_EW'][i]
            flags = galdict['flag_EW'][i]
            rrvir = maintab['rrvir'][i]
            qsoname = maintab['qsoname'][i]
            color = colors[i]
            for j,sis in enumerate(ews):
                if flags[j] == 3:
                    plt.errorbar(rrvir, ews[j], yerr=sigews[j], uplims=True, fmt='.', barsabove=True,
                                 c=color, label=qsoname, markersize=14, elinewidth=4, ecolor='mediumslateblue',) 
                else:
                    plt.errorbar(rrvir, ews[j], yerr=sigews[j], fmt='.', c=color, capsize=10, capthick=2,
                                 label=qsoname, markersize=14, elinewidth=4, ecolor='mediumslateblue')

    elif EW == 'summed':
        sigsum_1 = []
        sigsum_3 = []
        ew1_sum = []
        ew3_sum = []

        for i,sys in enumerate(galdict['sysid']):
            ews = galdict['EW'][i]
            sigews = galdict['sig_EW'][i]
            sqsigs = [n**2 for n in sigews]
            flags = galdict['flag_EW'][i]
            ew_1 = []
            ew_3 = []
            sqsig_1 = []
            sqsig_3 = []
            for j,flag in enumerate(flags):
                if flag == 1:
                    ew_1.append(ews[j])
                    sqsig_1.append(sqsigs[j])
                elif flag == 3:
                    ew_3.append(ews[j])
                    sqsig_3.append(sqsigs[j])
            ew1_sum.append(np.sum(ew_1))
            sigsum_1.append(np.sum(sqsig_1))
            ew3_sum.append(np.sum(ew_3))
            sigsum_3.append(np.sum(sqsig_3))

        for m,sys in enumerate(galdict['sysid']):
            rrvir = maintab['rrvir'][m]
            qsoname = maintab['qsoname'][m]
            color = colors[m]
            ew1 = ew1_sum[m]
            ew3 = ew3_sum[m]
            sig1 = np.sqrt(sigsum_1[m])
            sig3 = np.sqrt(sigsum_3[m])
            if ew1 == 0:
                pass
            else:
                plt.errorbar(rrvir, ew1, yerr=sig1, fmt='.', c=color, capsize=10, capthick=2,
                             label=qsoname, markersize=14, elinewidth=4, ecolor='mediumslateblue')
            if ew3 == 0:
                pass
            else:
                plt.errorbar(rrvir, ew3, yerr=sig3, uplims=True, fmt='.', barsabove=True,
                             c=color, label=qsoname, markersize=14, elinewidth=4, ecolor='mediumslateblue',) 


                    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys())


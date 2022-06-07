""" This contains all of the functions for identifying lines in sightlines """

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


def addspec(complist,flags,specpath,return_lines=True):
    """
    This function takes in a component list that has been loaded using igmguesses and instantiates each line in an 
    identified absorption component. The function also assigns a spectrum file to each line so that the rest 
    equivalent width and aodm can be and are measured for each line.
    
    
    Parameters
    ----------
    complist : list
        This should be a list containing absorption components.
    specpath : string
        This is the string leading to the spectrum file associated with the absorption component list.
        
    Returns
    -------
    listoflines : list
        This is a list containing the instantiated absorption components from the original complist. The spectrum for
        the abslines is included, as well as the aodm and equivalent width measurements.
    
    """
    
    specobj = readspec(specpath)
    listoflines = []
    
    for i,comp in enumerate(complist):
        lines = comp._abslines
        flag = flags[i]
        if flag == 1:
            for line in lines:
                line.analy['spec'] = specobj
                line.measure_restew()
                line.measure_aodm()
                listoflines.append(line)
        elif flag == 3:
            for line in lines:
                line.analy['spec'] = specobj
                line.measure_restew()
                line.measure_aodm()
                sig = line.attrib['sig_EW']
                line.attrib['EW'] = 3 * sig
                line.attrib['flag_EW'] = 3
                listoflines.append(line)


    if return_lines:
        return listoflines
    else:
        return complist


def get_lines(ion):
    """
    This function acts as a lookup for the lines from an ion. If given a species it returns the lines within that
    species. Input and output should both be in string format.

    Parameters
    ----------
    ion : string
        This input is for the ion species being sought. Ex: 'HI' or 'OVI'

    Returns
    -------
    lines : list of strings
        This returned list contains the line names for a specified ion. Ex: 'HI 1025' and 'HI 1215' for 'HI'
    """

    if ion == 'HI':
        return ['HI 1215', 'HI 1025']
    elif ion == 'OVI':
        return ['OVI 1031', 'OVI 1037']
    else:
        return "Ion not present in lookup table."


def idz(complist,clusterz,name,clustname,ions=['HI', 'OVI'],vrange=0.005):
    """
    This function takes in a component list that has been loaded using igmguesses and identifies the different 
    absorption components that are within a specified velocity range of each other, vrange. It then measures the 
    difference between the central-most redshift associated with grouped absorption components and the redshift of 
    the identified cluster, which is the other necessary input parameter for the function. The function then 
    identifies the grouped absorption components with the minimum difference between the cluster redshift and their 
    redshift, which is taken to be the absorption system(s) associated with the cluster. The cluster's absorption 
    components are then returned as a list. If there is no absorption component within the specified velocity range 
    of the cluster redshift, an absorption compoent is instantiated and flagged as a statistically insignificant 
    Lyman alpha detection.
    
    
    Parameters
    ----------
    complist : list
        This should be a list containing absorption components.
    clusterz : float array or list
        This is the column from the maintable containing all of the cluster and QSO information that contains the 
        redshifts for the clusters.
    name : string array or list
        This array or list is the column from the maintable containing the name of the QSOs the complist is being 
        drawn from.
    clustname : string array or list
        This array or list points to the column in the maintable where the cluster's name is.
    ions : list of strings
        This list corresponds to the ionization species being looked for. Default are 'HI' and 'OVI'.
    vrange : float
        This value corresponds to the veloctiy range the clustering algorithm is searching over. The default value, 
        0.005, corresponds with a velocity range of 1500 km/s. The z = v/c relationship can be used here to determine
        what value to use for different velocity ranges.
        
    Returns
    -------
    newzidcomps : list
        This list is returned if there is no absorption component found within the specified velocity range from the
        redshift of the cluster. The list therefore contains one manually instantiated absorption component that is 
        statistically insignificant and has an equivalent width measurement of zero. 
    zidcomps : list
        This list is returned if a cluster was successfully identified and it contains the absorption components
        associated with that cluster.
    finalflags : list
        This list is of the corresponding flags for the associated zid list.
    
    """
    
    complist = np.array(complist)
    finalzcomps = []
    
    vrange_gclust = vrange
    vrange_clust = vrange / 2
    
    redshifts = [i.z for i in complist]
    ccs, labels = au.cluster1d(redshifts,vrange_clust)
    zdiff = np.abs(ccs - clusterz)
    zmin = np.argmin(zdiff)

    if vrange_gclust < zdiff[zmin]:
        print("No absorption component associated with QSO spectrum " + name + ' and ' + str(clustname) + '.')
        newcomplist = complist.tolist()
        for ion in ions:
            lines = get_lines(ion)
            lines_set = []
            for linename in lines:
                line = AbsLine(linename, z = clusterz)
                line.limits.set([-50.,50.] * u.km / u.s)
                lines_set.append(line)
            newcomp = AbsComponent.from_abslines(lines_set)
            newcomplist.append(newcomp)
        newredshifts = [i.limits.z for i in np.array(newcomplist)]
        newccs, newlabels = au.cluster1d(newredshifts,vrange)
        newzdiff = np.abs(newccs - clusterz)
        newzmin = np.argmin(newzdiff)
        newzidxs = np.array(np.where(newlabels==newzmin)[0])
        newcomplist = np.array(newcomplist)
        newzidcomps = newcomplist[newzidxs]
        flags = []
        for comp in newzidcomps:
            flags.append(3)
        return list(newzidcomps), flags
    elif vrange_gclust >= zdiff[zmin]:
        indxcomps = []
        flags = []
        finalflags = []
        zidxs = np.array([], dtype=int)
        for i in np.arange(0,len(zdiff)):
            if zdiff[i] <= vrange_gclust:
                indxcomps.append(i)
        for j in indxcomps:
            idxs = np.where(labels==j)[0]
            zidxs = np.append(zidxs, idxs)
        zidcomps = [complist[i] for i in zidxs] # this is list with all comps at cluster z
        names = []
        for comp in zidcomps:
            lines = comp._abslines
            flags.append(1)
            for line in lines:
                names.append(line.name)
        linenamelist = []
        for ion in ions:
            if ion == 'HI':
                pass
            else:
                lines = get_lines(ion)
                for linename in lines:
                    linenamelist.append(linename)
                    missing_lines = []
                    if linename not in names:
                        print(linename + " not in identified lines for " + name + ' and ' + str(clustname) + '.')
                        line = AbsLine(linename, z=clusterz)
                        line.limits.set([-50., 50.] * u.km / u.s)
                        missing_lines.append(line)
                    else:
                        for comp in zidcomps:
                            compz = comp.zcomp
                            deltaz = np.abs(compz - clusterz)
                            if deltaz <= vrange_gclust:
                                finalzcomps.append(comp)
                                finalflags.append(1)
                        return finalzcomps, finalflags
                    newcomp = AbsComponent.from_abslines(missing_lines)
                    zidcomps.append(newcomp)
                    flags.append(3)
                for i,comp in enumerate(zidcomps):
                    compz = comp.zcomp
                    deltaz = np.abs(compz - clusterz)
                    if deltaz <= vrange_gclust:
                        finalzcomps.append(comp)
                        finalflags.append(flags[i])
                return finalzcomps, finalflags





def findline(idzlist, ion):
    """
    This function takes a component list that has already had the absorption system of the cluster identified
    and will return the rest frame equivalent width and aodm meaurements for a specific absorption line. The 
    absorption line being sought should be input as a string. For example, Lyman alpha would be 'HI 1215'.
    
    
    Parameters
    ----------
    idzlist : list
        This list is the list output from the idz function, which contains the identified absorption components for
        the cluster associated with a sightline.
    linename : string
        This string contains the name of the absorption line the user is wanting to identify and pull from the 
        absorption component list. The line naming convention must match that of the linetools package. For example,
        Lyman alpha would be 'HI 1215', and is the default line sought.
    
    Returns
    -------
    idlines : list
        This list returns a new list that contains only the specified absorption lines from linename. 
        
    Raises
    ------
    ValueError
        Either number of lines returned does not match number of lines present or line not present
    
    """

    idlines = []
    count = 0

    for comp in idzlist:
        lines = comp._abslines
        for line in lines:
            if ion in line.name:
                idlines.append(line)

    for comp in idzlist:
        lines = comp._abslines
        for line in lines:
            if ion in line.name:
                count = count + 1

    if len(idlines) == count:
        return idlines
    else:
        raise ValueError('Either number of lines returned does not match number of lines present or line is not present')



def comps(maintab, vrange=0.005):
    complist = igmguesses.from_igmguesses_to_complist(maintab['qsoname'] + '/IGM_model.json')
    idzcomp, flags = idz(complist, maintab['zclust'], maintab['qsoname'], maintab['clustname'], vrange)
    clist = addspec(idzcomp, maintab['qsoname'] + '/' + maintab['qsoname'] + '_cont.fits', return_lines=False)

    return clist


def mainfunc(maintab, linename='HI 1215', vrange=0.005, abscomp=False, SN=14):
    """
    This function uses the addspec, idz, and findline functions to create a dict of absorption components that
    corresponds to

    Parameters
    ----------
    maintab : table
        This table
    linename : string
        This string contains the name of the absorption line the user is wanting to identify and pull from the
        absorption component list. The line naming convention must match that of the linetools package. For example,
        Lyman alpha would be 'HI 1215', and is the default line sought.

    Returns
    -------
    idlines : list
        This list returns a new list that contains only the specified absorption lines from linename.

    Raises
    ------
    ValueError
        Either number of lines returned does not match number of lines present or line not present

    """

    sysids = []
    ews = []
    sigews = []
    flagews = []
    identifier = []
    abscomps = []
    lines = []

    for i in np.arange(0, len(maintab), 1):
        if maintab['qsoSN'][i] < SN:
            pass
        else:
            complist = igmguesses.from_igmguesses_to_complist(maintab['qsoname'][i] + '/IGM_model.json')
            idzcomp, flags = idz(complist, maintab['zclust'][i], maintab['qsoname'][i], maintab['clustname'][i])
            linelist = addspec(idzcomp, flags, maintab['qsoname'][i] + '/' + maintab['qsoname'][i] + '_cont.fits')
            lyalphas = findline(idzcomp, linename)
            if maintab['qsoname'][i] == 'FIRST-J020930.7-043826':
                if maintab['clustname'][i] == 33547:
                    tew = [0]
                    tsigew = [float(ll.attrib['sig_EW'].value) for ll in lyalphas]
                    tflag = [3]
            elif maintab['qsoname'][i] == 'PG1222+216':
                if maintab['clustname'][i] == 82403:
                    tew = [0]
                    tsigew = [float(ll.attrib['sig_EW'].value) for ll in lyalphas]
                    tflag = [3]
            else:
                tew = [float(ll.attrib['EW'].value) for ll in lyalphas]
                tsigew = [float(ll.attrib['sig_EW'].value) for ll in lyalphas]
                tflag = [int(ll.attrib['flag_EW']) for ll in lyalphas]
            ids = maintab['qsoname'][i] + '_' + str(maintab['clustname'][i])

            # if tew[0] < 0:
            #     tew[0] = np.abs(tew[0])
            sysids.append(int(i))
            ews.append(tew)
            sigews.append(tsigew)
            flagews.append(tflag)
            identifier.append(ids)
            abscomps.append(lyalphas)
            lines.append(complist)

    if abscomp == True:
        measdict = {'sysid': sysids, 'EW': ews, 'sig_EW': sigews, 'flag_EW': flagews, 'identifier': identifier,
                    'abscomp': abscomps, }
    elif abscomp == False:
        measdict = {'sysid': sysids, 'EW': ews, 'sig_EW': sigews, 'flag_EW': flagews, 'identifier': identifier, }

    return measdict


def workingmainfunc(maintab, linename='HI 1215', ions=['HI', 'OVI'], vrange=0.005, abscomp=False, SN=14):
    """
    This function uses the addspec, idz, and findline functions to create a dict of absorption components that 
    corresponds to
    
    Parameters
    ----------
    maintab : table
        This table
    linename : string
        This string contains the name of the absorption line the user is wanting to identify and pull from the 
        absorption component list. The line naming convention must match that of the linetools package. For example,
        Lyman alpha would be 'HI 1215', and is the default line sought.
    
    Returns
    -------
    idlines : list
        This list returns a new list that contains only the specified absorption lines from linename. 
        
    Raises
    ------
    ValueError
        Either number of lines returned does not match number of lines present or line not present
    
    """
    
    sysids = []
    ews=[]
    sigews=[]
    flagews = []
    identifier = []
    abscomps = []
    lines = []
    masterdict = dict()

    for i in np.arange(0,len(maintab),1):
        if maintab['qsoSN'][i] < SN:
            pass
        else:
            complist = igmguesses.from_igmguesses_to_complist(maintab['qsoname'][i] + '/IGM_model.json')
            idzcomp = idz(complist,maintab['zclust'][i],maintab['qsoname'][i],maintab['clustname'][i],ions,vrange)
            linelist = addspec(idzcomp, maintab['qsoname'][i] + '/' + maintab['qsoname'][i] + '_cont.fits', maintab['clustname'][i])
            measdict = dict()
            for ion in ions:
                ion_lines = findline(idzcomp,ion)
                tew = [float(ll.attrib['EW'].value) for ll in ion_lines]
                tsigew = [float(ll.attrib['sig_EW'].value) for ll in ion_lines]
                tflag = [int(ll.attrib['flag_EW']) for ll in ion_lines]
                ids = maintab['qsoname'][i] + '_' + str(maintab['clustname'][i])
                sysids.append(int(i))
                ews.append(tew)
                sigews.append(tsigew)
                flagews.append(tflag)
                identifier.append(ids)
                abscomps.append(ion_lines)
                lines.append(complist)

                if abscomp == True:
                    measdict = {'sysid': sysids, 'EW': ews, 'sig_EW': sigews, 'flag_EW': flagews, 'identifier': identifier,
                            'abscomp': abscomps,}
                elif abscomp == False:
                    measdict = {'sysid': sysids, 'EW': ews, 'sig_EW': sigews, 'flag_EW': flagews, 'identifier': identifier,}

            return measdict



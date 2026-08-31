import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt

def WriteDatasetsToCsv(annotationFileDir,trainingDir,outputDir,galNames,Nsnaps,inclinations,position_angles,simulationSuites,DoFaceOnProjections=False,paramString="MassFlux",includeAllRotations=True,useDenoisedSpectra=False,useTimeAveragedAnnotations=False,includeAllFlips=True):
    """Scan a directory tree of per-galaxy simulation outputs (from VeryObservableFIRE-style
    training data) and write a CoNNGaFit-compatible annotations CSV: one row per
    (galaxy, snapshot, inclination, position angle, rotation, flip) combination that has both
    an image and annotation file on disk, listing the image path followed by its flattened
    label values. Also saves a quick-look PNG of each annotation map under outputDir.

    annotationFileDir: output CSV path to write.
    trainingDir: root directory containing per-galaxy simulation output subdirectories.
    outputDir: directory the per-galaxy quick-look annotation PNGs are saved under.
    galNames: list of galaxy names (subdirectory names under trainingDir).
    Nsnaps / inclinations / position_angles: lists of snapshot numbers / inclination angles
        (degrees) / position angles (degrees) to include - the full cartesian product is
        scanned for each galaxy.
    simulationSuites: per-galaxy suite tag (parallel to galNames); a galaxy is only included
        if its suite is 'all' or 'cr'.
    DoFaceOnProjections: if True, look for annotations under a fixed "i90" (face-on)
        subdirectory instead of the per-inclination directory structure.
    paramString: which label type to write ("MassFlux", "inclination", "sMassFluxCurve", or
        any other custom tag), used to build both the annotation filename and the output CSV
        naming.
    includeAllRotations / includeAllFlips: whether to also include the 90/180/270-degree
        rotated and left-right-flipped copies of each image as additional distinct rows
        (basic data augmentation baked into the CSV rather than done at train time).
    useDenoisedSpectra / useTimeAveragedAnnotations: whether to look for the denoised-spectra
        ("_dn") and time-averaged-annotation ("_timeAvg") filename variants.
    """


    if includeAllRotations:rotationStrings=['','_r90','_r180','_r270']
    else: rotationStrings=['']

    #if includeAllFlips:flipStrings=['','_lr','_ud','_lr_ud']
    if includeAllFlips:flipStrings=['','_lr'] # Other flips are degenerate with rotations
    else: flipStrings=['']

    if paramString=="MassFlux":
        paramString2 = "MF"
    elif paramString=="inclination":
        paramString2 = "inc"
    elif paramString=="sMassFluxCurve":
        paramString2="sMF1d"
    else:
        paramString2 = paramString
    fid = open(annotationFileDir,'w')
    i=0
    galCount=-1
    
    dnStr=''
    taStr=''
    if useDenoisedSpectra:
        dnStr='_dn'
    if useTimeAveragedAnnotations:
        taStr='_timeAvg'
    
    itr=-1
    for gal in galNames:
        galCount+=1
        for Nsnap in Nsnaps:
            for inclination in inclinations:
              for position_angle in position_angles:
               for rotation in rotationStrings:
                for flip in flipStrings:
                  if simulationSuites[galCount]=='all' or simulationSuites[galCount]=='cr':
                    itr+=1
                    suiteStr = ''
                    iStr=str(inclination)
                    paStr = str(position_angle)
                    Nsnapstr =str(Nsnap)
                    actualImageName = gal+suiteStr+"_cr700_i"+iStr+"_pa"+paStr+"_"+Nsnapstr+"_image_04172023_fullSpectra"+rotation+dnStr+flip+".hdf5"
                    
                    if DoFaceOnProjections:
                        annotationDir = trainingDir+"i90/training/training_annotations_"+paramString+"_i90_"+gal+suiteStr+"_"+paramString2+"_"+Nsnapstr+rotation+taStr+flip+".hdf5"
                    else:
                        annotationDir = trainingDir+gal+"/vof_outputs/i"+iStr+"/training/training_annotations_"+paramString+"_i"+iStr+"_pa"+paStr+"_"+gal+suiteStr+"_"+paramString2+"_"+Nsnapstr+rotation+taStr+flip+".hdf5"

                    imageDir = trainingDir + gal+"/vof_outputs/i"+iStr+"/training/"+actualImageName

                    try:
                        hf=h5py.File(annotationDir,'r')
                        hfTmp = h5py.File(imageDir,'r')
                        hfTmp.close()
                        success=1
                        print("Opened ",annotationDir)
                    except:
                        success=0
                        print("Warning, could not open ",annotationDir)

                    if success:
                      annotation = np.array(hf['annotation'])
                      if np.size(np.shape(annotation))>1: annotation=annotation.flatten()
                      imageName = np.array(hf['imageName']).astype(str)

                      splitName=np.char.split(imageName,sep="/")
                      plt.figure()
                      vmax = np.max(np.abs(annotation))
                      vmin=-vmax
                      plt.imshow(np.reshape(annotation,(600,600)),vmin=vmin,vmax=vmax,cmap='seismic')
                      plt.savefig(os.path.join(outputDir,gal+"_i"+str(inclination)+'_pa'+str(position_angle)+'_'+rotation+'_'+flip+'.png'))
                      plt.close()

                      if i==0:
                        i+=1
                        fid.write(imageDir)
                        for j in range(0,np.size(annotation)):
                          fid.write(','+str(annotation[j]))
                      else: i+=1

                      fid.write('\n'+imageDir)
                      for j in range(0,np.size(annotation)):
                          fid.write(','+str(annotation[j]))
                      hf.close()
    fid.close()


####Run as: python WriteDatasetsToCsv_HiResTests.py [options]
####  Run with --help to see all options (simulation/output directories, output CSV name).
####  The galaxy/snapshot/inclination/position-angle lists below are left as in-file
####  parameters (like the TrainModel scripts' hyperparameters) rather than CLI flags, since
####  they describe which simulations to scan rather than where things live on disk.

def parse_args():
    parser = argparse.ArgumentParser(description="Scan a directory tree of FIRE-2 simulation outputs and write a CoNNGaFit-compatible training annotations CSV.")
    parser.add_argument('--sim-dir', default='/Volumes/wde4tb/simulation_snapshots/fire-2/',
                         help="Root directory containing per-galaxy simulation output subdirectories. Default: %(default)s")
    parser.add_argument('--output-dir', default='/Volumes/wde4tb/GalfitClean/training_datasets/',
                         help="Directory the output annotations CSV and quick-look PNGs are written to. Default: %(default)s")
    parser.add_argument('--output-name', default='training_annotations_HiRes_MassFlux',
                         help="Base name (no extension) for the output annotations CSV. Default: %(default)s")
    parser.add_argument('--use-denoised-spectra', action='store_true',
                         help="Look for the denoised-spectra ('_denoised') filename variant.")
    parser.add_argument('--use-time-averaged-annotations', action='store_true',
                         help="Look for the time-averaged-annotation ('_timeAvg') filename variant.")
    return parser.parse_args()

args = parse_args()

outputDir = args.output_dir
trainingDir = args.sim_dir

useDenoisedSpectra = args.use_denoised_spectra
useTimeAveragedAnnotations = args.use_time_averaged_annotations

dnStr=''
taStr=''
if useDenoisedSpectra: dnStr="_denoised"
if useTimeAveragedAnnotations: taStr = "_timeAvg"

### Only Good Galaxies ###
galNames = ['m12i','m12f','m12b','m12m']
suites = ['cr','cr','cr','cr']
Nsnaps = [600]
inclinations = [50]
position_angles = [0,45,90,135,180,225,270,315]

annotationFileDir = os.path.join(outputDir, args.output_name+dnStr+taStr+".csv")
WriteDatasetsToCsv(annotationFileDir,trainingDir,outputDir,galNames,Nsnaps,inclinations,position_angles,suites,useDenoisedSpectra=useDenoisedSpectra,useTimeAveragedAnnotations=useTimeAveragedAnnotations)
outputDir = '/Volumes/wde4tb/GalfitClean/training_datasets/'
trainingDir = '/Volumes/wde4tb/simulation_snapshots/fire-2/'

useDenoisedSpectra=False
useTimeAveragedAnnotations=False

dnStr=''
taStr=''
if useDenoisedSpectra: dnStr="_denoised"
if useTimeAveragedAnnotations: taStr = "_timeAvg"

### Only Good Galaxies ###
galNames = ['m12i','m12f','m12b','m12m']
suites = ['cr','cr','cr','cr']
Nsnaps = [600]
inclinations = [50]
position_angles = [0,45,90,135,180,225,270,315]

annotationFileDir = outputDir + "training_annotations_HiRes_MassFlux"+dnStr+taStr+".csv"
WriteDatasetsToCsv(annotationFileDir,trainingDir,galNames,Nsnaps,inclinations,position_angles,suites,useDenoisedSpectra=useDenoisedSpectra,useTimeAveragedAnnotations=useTimeAveragedAnnotations)


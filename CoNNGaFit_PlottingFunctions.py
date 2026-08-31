import numpy as np
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats as stats
import h5py

Sim2PhysicalUnits_MassFlux = (2/np.pi) * 1/(3.086*np.power(10.,16.)) * (3.154*np.power(10.,7.)) * np.power(10,10) #1/pixel_res * kpc2km * s2yr * unit mass to solar masses

#### Plotting Functions ####
def MakeCompImage(data_pred,data_input,data_image,output,Nsnap,Sim2PhysicalUnits=Sim2PhysicalUnits_MassFlux,paramLabel='Radial Mass Flux',unitLabel=r'[M$_{\odot}$ yr$^{-1}$]',binOp="sum"):
    """Save a 4-panel diagnostic figure (input map, prediction map, radially-binned
    input/prediction curves, and the summed input datacube) for one snapshot in a batch.

    data_pred / data_input: (nBatch, model_size) flattened prediction/label maps, or
        (nBatch, 1, npix, npix) already-2-d maps - both shapes are handled (see the
        try/except below, which falls back to the already-2-d case if the flat reshape fails).
    data_image: (nBatch, nSpec, nX, nY) input datacube batch, summed over the spectral axis
        for the bottom-right panel.
    output: path to save the PNG to.
    Nsnap: index of the sample within the batch to plot.
    Sim2PhysicalUnits: scalar multiplied into the data before plotting, to convert from
        simulation units to physical units (see Sim2PhysicalUnits_MassFlux above).
    paramLabel/unitLabel: axis/colorbar labels for the parameter being plotted.
    binOp: statistic passed to scipy.stats.binned_statistic_dd for the radial profile
        ("sum" for mass flux, "mean" for velocity/inclination, etc.).
    """
    try:
        image_pred = np.zeros((np.shape(data_pred)[1]))
        image_pred[:] = data_pred[Nsnap,:]

        image_input = np.zeros((np.shape(data_input)[1]))
        image_input[:] = data_input[Nsnap,:]

        npix = int(np.round(np.sqrt(np.size(image_pred))))
        image_pred = np.reshape(image_pred,[npix,npix])
        image_input = np.reshape(image_input,[npix,npix])
    except: #data_pred/data_input were already (nBatch,1,npix,npix) rather than flattened
        image_pred = data_pred[Nsnap,0,:,:]
        image_input = data_input[Nsnap,0,:,:]
        npix = int(np.round(np.sqrt(np.size(image_pred))))


    plt.figure()
    
    plt.subplot(2,2,4)
    imageToPlot = np.sum(data_image[Nsnap,:,:,:],axis=0)
    imageToPlot+=np.abs(np.min(imageToPlot))+1
    plt.imshow(imageToPlot,norm=LogNorm(),cmap='inferno')
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])
    
    plt.subplot(2,2,3)
    centerIndex = [(npix-1.)/2.,(npix-1.)/2.]
    indices = np.indices(np.shape(image_pred))
    rmag = np.sqrt( np.power(indices[0,:,:]-centerIndex[0] , 2) + np.power(indices[1,:,:] -centerIndex[1], 2) )
    
    nBins = 20
    
    binRange=[0,np.max(rmag)]
    binned_data_pred,binedge,binnum = stats.binned_statistic_dd(rmag.flatten(),image_pred.flatten(),binOp,nBins,range=[binRange])
    binned_data_input,binedge,binnum = stats.binned_statistic_dd(rmag.flatten(),image_input.flatten(),binOp,nBins,range=[binRange])
    maxPlot = np.max([-np.min(binned_data_pred),-np.min(binned_data_input),np.max(binned_data_pred),np.max(binned_data_input)])
    plt.plot(np.linspace(0,np.max(rmag),nBins) , binned_data_input* Sim2PhysicalUnits , 'k',lw = 2)
    plt.plot(np.linspace(0,np.max(rmag),nBins) , binned_data_pred * Sim2PhysicalUnits , 'r--' ,lw = 2)
    plt.legend(['Input','Prediction'])
    plt.plot(np.linspace(0,np.max(rmag),nBins) , binned_data_pred * 0 , 'k--' ,lw = 1)
    plt.xlabel('Radius')
    plt.ylabel(paramLabel+" "+unitLabel)
    plt.ylim([-maxPlot*1.05* Sim2PhysicalUnits,maxPlot*1.05* Sim2PhysicalUnits])
    
    
    vmax = np.max( [-np.min(image_pred) , np.max(image_pred), -np.min(image_input) , np.max(image_input)])
    vmin=-vmax
    plt.subplot(2,2,1)
    plt.imshow(image_input * Sim2PhysicalUnits,cmap='seismic',vmin=vmin* Sim2PhysicalUnits,vmax=vmax* Sim2PhysicalUnits)
    plt.title(paramLabel+" Input")
    #plt.colorbar();
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])
    plt.subplot(2,2,2)
    plt.imshow(image_pred * Sim2PhysicalUnits,cmap='seismic',vmin=vmin* Sim2PhysicalUnits,vmax=vmax* Sim2PhysicalUnits)
    plt.colorbar(label = unitLabel);
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])
    plt.title(paramLabel+" Prediction")
    plt.savefig(output)
    plt.close()
    
def MakeImage(data,output,vmin=None,vmax=None,targetShape=None,Sim2PhysicalUnits=Sim2PhysicalUnits_MassFlux,paramLabel='Radial Mass Flux',unitLabel=r'[M$_{\odot}$ yr$^{-1}$]'):
    """Reshape a single flattened prediction into a 2-d map, save it as a PNG (with a
    physical-units colorbar) and an accompanying hdf5 file.

    data: 1-d array of length npix*npix (a single sample's flattened prediction) - note the
        (1, N) shape convention (data[1] indexing) mirrors the batched functions above but
        here data is expected to already be for one sample.
    output: path prefix; writes output+"projectionMap.png"/".hdf5".
    vmin/vmax: fixed color limits; if either is None, both default to symmetric limits
        derived from the data's own min/max.
    targetShape: (nY, nX) shape to reshape the flat data into; pass this explicitly as a
        keyword when calling (not positionally - the 3rd positional argument is vmin, not
        targetShape).
    """
    image = np.zeros((np.shape(data)[1]))
    image[:] = data[:]
    
    if targetShape is None:
        npix = int(np.round(np.sqrt(np.size(image))))
        targetShape=[npix,npix]
        
    image = np.reshape(image,targetShape) #make into 2d image

    plt.figure()
    
    if vmax is None or vmin is None:
        vmax = np.max( [-np.min(image) , np.max(image)]) # default to symmetric limits
        vmin=-vmax
    
    #Plot in units of Solar Masses per year
    plt.imshow(image*Sim2PhysicalUnits,cmap='seismic',vmin=vmin*Sim2PhysicalUnits,vmax=vmax*Sim2PhysicalUnits)
    plt.colorbar(label = paramLabel+' '+unitLabel);
    plt.savefig(output+"projectionMap.png")
    plt.close()
    
    #Save hdf5 for generating figures
    hf=h5py.File(output+"projectionMap.hdf5",'w')
    hf.create_dataset('image',data=image)
    hf.close()

def CreateBasicPlots(data,output,targetShape=None,nBins=20,Sim2PhysicalUnits=Sim2PhysicalUnits_MassFlux,paramLabel='Radial Mass Flux',unitLabel=r'[M$_{\odot}$ yr$^{-1}$]'):
    """Reshape a single flattened prediction into a 2-d map, azimuthally bin it into a
    radial profile by galactocentric pixel radius, and save the profile as a PNG + hdf5.

    data: 1-d array of length npix*npix for one sample (see MakeImage above for the shape
        convention).
    output: path prefix; writes output+"radialPlot.png"/".hdf5".
    targetShape: (nY, nX) shape to reshape data into; defaults to a square inferred from
        len(data).
    nBins: number of radial bins to sum the map into.
    """
    image = np.zeros((np.shape(data)[1]))
    image[:] = data[:]
    if targetShape is None:
        npix = int(np.round(np.sqrt(np.size(image))))
        targetShape=[npix,npix]
    
    npix=targetShape[0]
    image = np.reshape(image,targetShape) #make into 2d image
    
    ## Calculate galactocentric radius of each pixel
    centerIndex = [(npix-1.)/2.,(npix-1.)/2.]
    indices = np.indices(np.shape(image))
    rmag = 2*np.sqrt( np.power(indices[0,:,:]-centerIndex[0] , 2) + np.power(indices[1,:,:] -centerIndex[1], 2) )
    
    ##Calculate total mass flux as function of radius
    binRange=[0,np.max(rmag)]
    binned_data,binedge,binnum = stats.binned_statistic_dd(rmag.flatten(),image.flatten(),"sum",nBins,range=[binRange])
    
    #Default to symmetric y-limits
    vmax = np.max( [-np.min(binned_data* Sim2PhysicalUnits) , np.max(binned_data* Sim2PhysicalUnits)])*1.05
    vmin=-vmax
    
    #Plot in units of Solar Masses per year
    plt.figure()
    plt.plot(np.linspace(0,np.max(rmag),nBins) , binned_data* Sim2PhysicalUnits , 'k', lw = 2.5) #plot azimuthal average
    plt.plot(np.linspace(0,np.max(rmag),nBins) , binned_data*0 , 'k--', lw = 1.5) #plot zero line
    plt.xlabel('Radius [kpc]')
    plt.ylabel(paramLabel+' '+unitLabel)
    plt.ylim([vmin,vmax])
    plt.savefig(output+"radialPlot.png")
    plt.close()
    
    #Save hdf5 for generating figures
    hf=h5py.File(output+"radialPlot.hdf5",'w')
    hf.create_dataset('binned_data',data=binned_data)
    hf.create_dataset('rPlot',data=np.linspace(0,np.max(rmag),nBins))

    hf.close()


def RV2coeff(dataList):
    """Compute the pairwise RV coefficient matrix (a multivariate generalization of squared
    correlation, in [0, 1]) between a list of 2-d arrays of the same shape - used by
    MakeCorrelationPlot below to score 2-d map similarity between prediction and truth.
    Returns a (len(dataList), len(dataList)) matrix C where C[i,j] is the RV coefficient
    between dataList[i] and dataList[j].
    """
    # First compute the scalar product matrices for each data set X
    scalArrList = []

    for arr in dataList:
        scalArr = np.dot(arr, np.transpose(arr))
        diego = np.diag(np.diag(scalArr))
        scalArrMod = scalArr - diego
        scalArrList.append(scalArrMod)

    # Now compute the 'between study cosine matrix' C
    C = np.zeros((len(dataList), len(dataList)), float)

    for index, element in np.ndenumerate(C):
        nom = np.trace(
            np.dot(np.transpose(scalArrList[index[0]]),
                      scalArrList[index[1]]))
        denom1 = np.trace(
            np.dot(np.transpose(scalArrList[index[0]]),
                      scalArrList[index[0]]))
        denom2 = np.trace(
            np.dot(np.transpose(scalArrList[index[1]]),
                      scalArrList[index[1]]))
        Rv = nom / np.sqrt(denom1 * denom2)
        C[index[0], index[1]] = Rv

    return C
    
def MakeCorrelationPlot(pred_image,input_image,Nsnaps,output):
    """Score prediction-vs-truth agreement across a batch of snapshots and save a histogram
    of two correlation metrics, plus the raw per-snapshot values to an hdf5 file.

    pred_image / input_image: (Nsnaps, model_size) flattened prediction/label maps.
    Nsnaps: number of snapshots (rows) to score.
    output: path prefix; writes output+".png"/".pdf"/".hdf5".

    For each snapshot, reshapes to a 2-d npix x npix map and computes:
      corr2d - the RV coefficient (see RV2coeff) between the full 2-d prediction and truth maps.
      corr1d - the Pearson correlation between the two maps' radially-binned 1-d profiles.
    """
    tickSize=14
    labelSize=16
    titleSize=18
    cbarSize=14
    nameSize=14
    legendSize = 14
    
    npix=40

    corr1d = np.zeros((Nsnaps))
    corr2d = np.zeros((Nsnaps))
    for i in range(0,Nsnaps):
        pred_snap = pred_image[i,:]
        input_snap = input_image[i,:]


    
        npix = int(np.round(np.sqrt(np.size(pred_snap))))
        pred_snap = np.reshape(pred_snap,[npix,npix])
        input_snap = np.reshape(input_snap,[npix,npix])
        
        rv2_mat = RV2coeff([input_snap,pred_snap])
        corr2d[i] = rv2_mat[0,1]

        centerIndex = [(npix-1.)/2.,(npix-1.)/2.]
        indices = np.indices(np.shape(pred_snap))
        rmag = np.sqrt( np.power(indices[0,:,:]-centerIndex[0] , 2) + np.power(indices[1,:,:] -centerIndex[1], 2) )
    
        nBins = npix
        binRange=[0,np.max(rmag)]
        binned_pred,binedge,binnum = stats.binned_statistic_dd(rmag.flatten(),pred_snap.flatten(),"sum",nBins,range=[binRange])
        binned_input,binedge,binnum = stats.binned_statistic_dd(rmag.flatten(),input_snap.flatten(),"sum",nBins,range=[binRange])
        
        correlationMatrix1d = np.corrcoef(binned_pred,binned_input)
        corr1d[i] = correlationMatrix1d[1][0]
        
    plt.figure()
    size = 60
    alpha=0.7
    predColor = 'r'
    inputColor = 'k'
    
    xplot = np.linspace(0,np.size(corr1d),np.size(corr1d))


    nBins = 20
    alpha = 0.7
    plt.hist(corr1d, bins=nBins,range=[-1,1],histtype='step',color='k',label='1d Corr.',alpha=alpha,linewidth=2)
    plt.hist(corr2d, bins=nBins,range=[-1,1],histtype='step',color='r',label='2D Corr.',alpha=alpha,linewidth=2)
    
    plt.xlim([-1,1])    

    plt.legend(fontsize=legendSize,loc='upper left')
    plt.ylabel('Snapshots',fontsize=labelSize)
    plt.xlabel('Correlation',fontsize=labelSize)
    plt.xticks(fontsize=tickSize)
    plt.yticks(fontsize=tickSize)
    plt.tight_layout()
    plt.savefig(output+".png",bbox_inches='tight')
    plt.savefig(output+".pdf",bbox_inches='tight')
    
    plt.close()        
    
    hf = h5py.File(output+".hdf5",'w')
    hf.create_dataset("corr1d",data=corr1d)
    hf.create_dataset("corr2d",data=corr2d)
    hf.close()
    

    
def SaveHDF5(inputData,predData,output,Nsnap):
    """Reshape one snapshot's flattened prediction (and optionally its label) into a 2-d
    map and save both to an hdf5 file for later re-plotting.

    inputData: (nBatch, model_size) flattened label maps, or None to skip saving the label
        (e.g. when running inference with no ground truth available).
    predData: (nBatch, model_size) flattened prediction maps.
    output: full path to the hdf5 file to write (datasets 'input' and/or 'prediction').
    Nsnap: index of the sample within the batch to save.
    """
    predImage = np.zeros((np.shape(predData)[1]))
    predImage[:] = predData[Nsnap,:]
    npix = int(np.round(np.sqrt(np.size(predImage))))
    predImage = np.reshape(predImage,[npix,npix])
    
    hf = h5py.File(output,'w')
    if inputData is not None:
        inputImage = np.zeros((np.shape(inputData)[1]))
        inputImage[:] = inputData[Nsnap,:]
        npix = int(np.round(np.sqrt(np.size(inputImage))))
        inputImage = np.reshape(inputImage,[npix,npix])
        hf.create_dataset('input',data=inputImage)
    hf.create_dataset('prediction',data=predImage)
    hf.close()
    

def LoadNames(inputDir):
    """Extract a short, distinct name for each row of a CoNNGaFit annotations CSV, by taking
    the basename (without extension) of the image path in each line.

    inputDir: path to an annotations CSV (as produced for training/validation/testing, or an
        inference image list) - only the leading image-path portion of each line is used.
    Returns: a 0-indexed list of names, one per line/row, in file order - i.e. names[0]
        corresponds to the CSV's first data row. Callers must index this list starting at 0,
        not 1.
    """
    Nlines=0
    fid = open(inputDir,'r')
    names = []
    for line in fid:
        x=line.split("/")
        Nsplit = np.size(x)
        y=(x[Nsplit-1]).split('.') #get filename in directory
        if "_md" in line and not "_md" in y[0]:
            names.append(y[0]+"_md")
        else:
            names.append(y[0]) #ignore filetype
        
    fid.close()
    
    return names
    
    
    
    
    
    
    
    
def MakeCompImage_1d(data_pred,data_input,data_image,output,Nsnap,Sim2PhysicalUnits=Sim2PhysicalUnits_MassFlux,paramLabel='Radial Mass Flux',unitLabel=r'[M$_{\odot}$ yr$^{-1}$]',binOp="sum"):
    """1-d analog of MakeCompImage: save a 2-panel figure (summed input datacube, and the
    input/prediction radial curves already stored in data_pred/data_input) for one snapshot.

    data_pred / data_input: (nBatch, nRadialBins) already-radially-binned prediction/label
        curves (unlike MakeCompImage, these are not reshaped into a 2-d map here).
    data_image: (nBatch, nSpec, nX, nY) input datacube batch.
    output/Nsnap/Sim2PhysicalUnits/paramLabel/unitLabel: see MakeCompImage above.
    """
    image_pred = np.zeros((np.shape(data_pred)[1]))
    image_pred[:] = data_pred[Nsnap,:]
    
    image_input = np.zeros((np.shape(data_input)[1]))
    image_input[:] = data_input[Nsnap,:]
    
    #npix = int(np.round(np.sqrt(np.size(image_pred))))
    #image_pred = np.reshape(image_pred,[npix,npix])
    #image_input = np.reshape(image_input,[npix,npix])

    plt.figure()
    
    plt.subplot(122)
    imageToPlot = np.sum(data_image[Nsnap,:,:,:],axis=0)
    imageToPlot+=np.abs(np.min(imageToPlot))+1
    plt.imshow(imageToPlot,norm=LogNorm(),cmap='inferno')
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])
    
    plt.subplot(121)

    rmag = np.linspace(0,30,np.size(data_pred[Nsnap,:]))
    
    nBins = 20

    maxPlot = np.max([-np.min( np.array(data_pred[Nsnap,:])),-np.min(np.array(data_input[Nsnap,:])),np.max( np.array(data_pred[Nsnap,:])),np.max(np.array(data_input[Nsnap,:]) )])
    plt.plot(rmag , np.array(data_input[Nsnap,:]) * Sim2PhysicalUnits , 'k',lw = 2)
    plt.plot(rmag , np.array(data_pred[Nsnap,:]) * Sim2PhysicalUnits , 'r--' ,lw = 2)
    plt.plot(rmag,rmag*0,'k--',lw=1)
    plt.legend(['Input','Prediction'])
    plt.xlabel('Radius')
    plt.ylabel(paramLabel+" "+unitLabel)
    plt.ylim([-maxPlot*1.05* Sim2PhysicalUnits,maxPlot*1.05* Sim2PhysicalUnits])
    
    

    plt.savefig(output)
    plt.close()
    
    
def SaveHDF5_1d(inputData,predData,output,Nsnap):
    """1-d analog of SaveHDF5: save one snapshot's already-1-d input/prediction radial
    curves to an hdf5 file. Note Nsnap is currently unused - the full inputData/predData
    arrays are saved as-is rather than being indexed down to a single snapshot; pass
    already-sliced 1-d arrays for a single sample.
    """
    hf = h5py.File(output,'w')
    hf.create_dataset('input',data=inputData)
    hf.create_dataset('prediction',data=predData)
    hf.close()
    
    
import numpy as np
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats as stats
import h5py

Sim2PhysicalUnits_MassFlux = (2/np.pi) * 1/(3.086*np.power(10.,16.)) * (3.154*np.power(10.,7.)) * np.power(10,10) #1/pixel_res * kpc2km * s2yr * unit mass to solar masses

#### Plotting Functions ####
def MakeCompImage(data_pred,data_input,data_image,output,Nsnap,Sim2PhysicalUnits=Sim2PhysicalUnits_MassFlux,paramLabel='Radial Mass Flux',unitLabel='[M$_{\odot}$ yr$^{-1}$]',binOp="sum"):
    image_pred = np.zeros((np.shape(data_pred)[1]))
    image_pred[:] = data_pred[Nsnap,:]
    
    image_input = np.zeros((np.shape(data_input)[1]))
    image_input[:] = data_input[Nsnap,:]
    
    npix = int(np.round(np.sqrt(np.size(image_pred))))
    image_pred = np.reshape(image_pred,[npix,npix])
    image_input = np.reshape(image_input,[npix,npix])

    plt.figure()
    
    plt.subplot(2,2,4)
    imageToPlot = np.sum(data_image[Nsnap,:,:,:],axis=0)
    imageToPlot+=np.abs(np.min(imageToPlot))+1
    plt.imshow(imageToPlot,norm=LogNorm(),cmap='inferno')
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])
    
    plt.subplot(2,2,3)
    centerIndex = [npix/2-1,npix/2-1]
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
    
def MakeImage(data,output,vmin=None,vmax=None,targetShape=None,Sim2PhysicalUnits=Sim2PhysicalUnits_MassFlux,paramLabel='Radial Mass Flux',unitLabel='[M$_{\odot}$ yr$^{-1}$]'):
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

def CreateBasicPlots(data,output,targetShape=None,nBins=20,Sim2PhysicalUnits=Sim2PhysicalUnits_MassFlux,paramLabel='Radial Mass Flux',unitLabel='[M$_{\odot}$ yr$^{-1}$]'):
    image = np.zeros((np.shape(data)[1]))
    image[:] = data[:]
    if targetShape is None:
        npix = int(np.round(np.sqrt(np.size(image))))
        targetShape=[npix,npix]
    
    npix=targetShape[0]
    image = np.reshape(image,targetShape) #make into 2d image
    
    ## Calculate galactocentric radius of each pixel
    centerIndex = [npix/2-1,npix/2-1]
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

        centerIndex = [npix/2-1,npix/2-1]
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
    inputImage = np.zeros((np.shape(inputData)[1]))
    inputImage[:] = inputData[Nsnap,:]
    npix = int(np.round(np.sqrt(np.size(inputImage))))
    inputImage = np.reshape(inputImage,[npix,npix])
    
    predImage = np.zeros((np.shape(predData)[1]))
    predImage[:] = predData[Nsnap,:]
    npix = int(np.round(np.sqrt(np.size(predImage))))
    predImage = np.reshape(predImage,[npix,npix])
    
    hf = h5py.File(output,'w')
    hf.create_dataset('input',data=inputImage)
    hf.create_dataset('prediction',data=predImage)
    hf.close()
    

def LoadNames(inputDir):
    ####Get a distinct name for each image to run inferences on
    Nlines=0
    fid = open(inputDir,'r')
    names = []
    for line in fid:
        x=line.split("\\")
        Nsplit = np.size(x)
        y=(x[Nsplit-1]).split('.') #get filename in directory
        names.append(y[0]) #ignore filetype
        
    fid.close()
    
    return names
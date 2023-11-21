from CoNNGaFit_UseModel import RunInferences
#### User Input ############################  

####Network to use
networkType = 'unet18'
####Address of csv file with list of images to to apply model to
imageList = "CoNNGaFitData/observations/inferencesToRun.csv"
####Directory to place outputs in
imageOutput = "CoNNGaFitData/observations/output_example"
############################################


RunInferences(networkType , dirName, imageOutput, imageList)

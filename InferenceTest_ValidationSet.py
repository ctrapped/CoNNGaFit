from CoNNGaFit_UseModel import RunInferences
#### User Input ############################  
sampleSuffix = 'finalSnapNoM12m'

trainingSetFilename = 'training_annotations_MassFlux_All_Inclinations_'+sampleSuffix+'.csv'
validationSetFilename = 'validation_annotations_MassFlux_All_Inclinations_'+sampleSuffix+'.csv'
testingSetFilename = 'test_annotations_MassFlux_All_Inclinations_'+sampleSuffix+'.csv'
outputFilebase = 'massFlux_'+sampleSuffix

trainingDir = 'CoNNGaFitData\\annotation_datasets\\'+trainingSetFilename
validationDir = 'CoNNGaFitData\\annotation_datasets\\'+validationSetFilename
testingDir = 'CoNNGaFitData\\annotation_datasets\\'+testingSetFilename


####Network to use
networkType = 'unet18'
####Address of csv file with list of images to to apply model to
imageList = validationDir
####Directory to place outputs in
imageOutput = "CoNNGaFitData/observations/validation_tests/"
############################################


RunInferences(networkType , imageOutput, imageList,saveLatentImages=True)

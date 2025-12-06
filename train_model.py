modelName = 'baseline'

args = {}
args['outputDir'] = '/home/andrew43/project/outputs/' + modelName
args['datasetPath'] = '/home/andrew43/project/competitionData/ptDecoder_ctc.pkl'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.05
args['lrEnd'] = 0.005
args['nUnits'] = 1024 #1024
args['nBatch'] = 15000
args['nLayers'] = 5 #5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256 #This is not editable!
args['dropout'] = 0.2
args['whiteNoiseSD'] = 1.2#0.8

args['constantOffsetSD'] = 0.2 #0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 3 #4
args['kernelLen'] = 24 #32
args['bidirectional'] =False
args['l2_decay'] = 1e-5

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
class BaseOptions():
    def initialize(self):
        self.dataroot = '/repo3/101_ObjectCategories/' # path to the dir of the dataset
        self.name = 'Mobilenetv2_default' # Name of the experiment
        self.batch_size =5  

class TrainOptions(BaseOptions):
    def __init__(self):

        BaseOptions.initialize(self)
        self.niter = 100
        self.input_c = 32 #input # of channel
        self.final_c = 1280 #last # of channel
        self.dropout = 0.2 #dropout rate
        self.momentum = 0.9 # momentum for the optimizer
        self.decay_optim = 0.00004 # weight decay for the optimizer
        self.lr = 0.045 #learning rate
        self.gamma = 0.98 #learning rate decay rate
        self.n_class = 101 # number of class for caltech 101 dataset
        self.input_size = 224 # input image size
        self.architecture =  [  # architecture of the mobilenet ;; paper table2
                    # t, c, n, s
                    [1, 16, 1, 1],
                    [6, 24, 2, 2],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1],
                ]
        self.linear = True # use linear bottleneck: True, otherwise False
        self.connect = True # Use residual connection: Ture, otherwise False
        self.use_gpu = True
        self.multi_gpu = True
        self.gpus = '2,3'

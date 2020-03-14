class Config:
    def __init__(self):
        self.num_class = 2
        self.batch_size = 8
        self.learning_rate = 0.0001
        self.max_epoch = 15
        self.size = [224,224]
        self.mode = "left"
        self.root = "F:/finish_multi/"
        self.x_min = 0
        self.y_min = 0
        self.heat_value = 175
        self.standard = 0.7
        self.xml_root = "F:/xml_txt_all/"

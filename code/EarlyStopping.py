class EarlyStopping:
    def __init__(self, tolerance = 5, min_delta = 0, counter = 0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = counter
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        # print(abs(validation_loss - train_loss))
        # print(self.min_delta)
        # print(abs(validation_loss - train_loss) > self.min_delta)

        if (abs(validation_loss - train_loss)> self.min_delta) :
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
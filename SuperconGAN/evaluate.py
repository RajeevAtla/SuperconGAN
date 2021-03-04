from sdv.evaluation import evaluate

class Evaluator():

    def __init__(self,
                synthetic_data,
                real_data):
        
        self.synthetic_data = synthetic_data
        self.real_data = real_data

    def evaluate_data(self,
                      aggregate = True,
                      metrics = None):
        
        evaluate(synthetic_data = self.synthetic_data,
                 real_data = self.real_data,
                 metrics = metrics,
                 aggregate = aggregate)

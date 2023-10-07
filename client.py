import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.algorithms.optimizers import COBYLA
from IPython.display import clear_output

from build_qnn import build_qnn



    


class Client():
    def __init__(self, client_id, 
                 train_x, train_y, 
                 test_x, test_y, 
                 q_num, local_ep,
                 save_path):
        self.client_id = client_id
        self.train_x = train_x  # Local dataset for this client
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

        self.save_path = save_path

        # Define a local model (you can customize this)
        self.local_model = build_qnn(q_num)
        self.local_weights = None

        self.local_epoch = local_ep
        self.objective_func_vals = []
        plt.rcParams["figure.figsize"] = (12, 6)

        self.set_classifier()

    
    def set_classifier(self):
        self.classifier = NeuralNetworkClassifier(
            self.local_model,
            optimizer=COBYLA(maxiter=self.local_epoch),  # Set max iterations here
            warm_start=True,
            callback=self.callback_graph
        )

    def callback_graph(self, weights, obj_func_eval):
        clear_output(wait=True)
        self.objective_func_vals.append(obj_func_eval)
        plt.title("Objective function value against iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(range(len(self.objective_func_vals)), self.objective_func_vals)
        plt.savefig(self.save_path + 'clt_Loss_{}.jpg'.format(self.client_id))

        # if len(objective_func_vals)%50 == 0:
        # np.save(weight_path + 'ckp_{}.npy'.format(len(objective_func_vals)), weights)


    def train_locally(self):
        print('local training index {}'.format(self.client_id))
        self.classifier.fit(self.train_x, self.train_y)

    def get_parameters(self):
        return self.classifier.weights
    

    def set_parameters(self, global_weights):
        self.classifier.initial_point = global_weights
        self.classifier._fit_result.x = global_weights
        # self.classifier = NeuralNetworkClassifier(
        #     self.local_model,
        #     optimizer=COBYLA(maxiter=self.local_epoch),  # Set max iterations here
        #     warm_start=True,
        #     initial_point=global_weights,
        #     callback=self.callback_graph
        # )
        self.local_weights = global_weights
        

    def evaluation(self):
        test_acc = np.round(100 * self.classifier.score(self.test_x, self.test_y), 2)
        
        return test_acc
    
    def get_localloss(self):
        return self.objective_func_vals


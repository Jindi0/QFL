import os
import argparse
from data_helper import load_dataset
import numpy as np
from client import Client
from util import init_log



def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='fedlearn_reg', help='task name')
    parser.add_argument('--client_n', type=int, default=10, help='the number of training samples')

    parser.add_argument('--local_epoch', type=int, default=10, help='the number of epoch during local training')
    parser.add_argument('--global_round', type=int, default=2000, help='the number of round during global training')

    # parser.add_argument('--samples', type=int, default=1000, help='the number of training samples')
    # parser.add_argument('--COBYLAiter', type=int, default=2000,help="the number of epochs")
    parser.add_argument('--train', type=bool, default=True, help='if perform training procedure')
    parser.add_argument('--test', type=bool, default=True, help='if perform test procedure')

    parser.add_argument('--q_num', type=int, default=7, help='the input size')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
   
    args = parser.parse_args()
    return args

args = args_parser()

savepath = './QFL/save_reg/'
if not os.path.exists(savepath):
    os.mkdir(savepath)
task_path = savepath + '{}/'.format(args.task)
if not os.path.exists(task_path):
    os.mkdir(task_path)
    os.mkdir(task_path + 'client/')
    os.mkdir(task_path + 'checkpoints/')





if __name__ == "__main__":
    client_data_train_x, client_data_train_y, client_data_test_x, client_data_test_y = load_dataset(args.client_n, args.q_num)

    args.client_n = 3

    f, sheet = init_log(args.client_n)

    # Create client objects
    clients = []
    for i in range(args.client_n):
        client = Client(client_id=i, 
                        train_x=client_data_train_x[i], 
                        train_y=client_data_train_y[i], 
                        test_x=client_data_test_x[i], 
                        test_y=client_data_test_y[i], 
                        q_num=args.q_num,
                        local_ep=args.local_epoch,
                        save_path=task_path+'client/')
        
        clients.append(client)

    

    # Federated learning
    for r in range(args.global_round):
        print('Global round: {}'.format(r))
        local_model = []
        for i in range(args.client_n):
            clients[i].train_locally()
            local_model.append(clients[i].get_parameters())
        local_model = np.array(local_model)

        global_model = np.mean(local_model, axis=0)
        np.save(task_path + 'checkpoints/global_ckp_{}.npy'.format(r), global_model)

        for i in range(args.client_n):
            clients[i].set_parameters(global_model)

            test_acc = clients[i].evaluation()
            sheet.write(r+1, i*2+2, test_acc)
            f.save(task_path + 'records.xls')

    # record local training loss
    for i in range(args.client_n):
        local_loss = clients[i].get_localloss()
        for j in range(len(local_loss)):
            sheet.write(j+1, args.client_n*2 +i+3, local_loss[j])
        f.save(task_path + 'records.xls')







    




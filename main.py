import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_num_threads(1)
import networkx as nx
import argparse

from utils import projection_order1, mask_from_order
from data_generator import data_generator, generate_DAG, generator_matrix
import opt
import time
from tqdm import tqdm
import pickle

def main(d, noise_type, graph_type, epochs, degree, train_size, test_size, seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)

    # Setting seeds for reproducibility
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    B_scale = 1.0 
    B_ranges = ((B_scale * -1.0, B_scale * -0.05), (B_scale * 0.05, B_scale * 1.0))

    # Generate DAG and matrix
    DAG = generate_DAG(d, graph_type, degree, B_ranges, hd=False, seed=seed)
    assert nx.is_directed_acyclic_graph(nx.DiGraph(DAG.B))
    B = torch.tensor(DAG.B).to(device)
    GM = generator_matrix(B)
    
    # Generate test and training data
    X_test = data_generator(generator_matrix=GM, bs=test_size, noise_type=noise_type, seed=seed).to(device)
    X = data_generator(generator_matrix=GM, bs=train_size, noise_type=noise_type, seed=seed).to(device)
    
    testing = False
    verbose = False

    def full_loss_(X, B):
        return (0.5 / X.size()[0]) * torch.square(X - X @ B).sum()
    
    true_order = projection_order1(B)
    optimal_test_loss = full_loss_(X_test, B)

    main_mask = torch.eye(d).to(device) == 0
    D0 = torch.zeros(d, d).to(device) * main_mask
    true_distance = (D0 - B).norm()
    distance = torch.ceil(true_distance)

    @torch.no_grad()
    def full_loss(D):
        return (0.5 / X_test.size()[0]) * torch.square(X_test - X_test @ D).sum()

    def loss(D, bs, mask):
        x = data_generator(GM, bs=bs).to(device)
        return 0.5 / bs * torch.square(x - x @ (D * mask)).sum()

    def quadratic_optimization(D, optimizer, num_iter, mask, loss=loss, full_loss=full_loss, bs=1, log_iter=100, testing=testing):
        losses_inside = []
        with torch.no_grad():
            D *= mask
        for i in range(num_iter):
            if i % log_iter == 0 and testing:
                losses_inside.append(full_loss(D).item())

            def closure():
                optimizer.zero_grad()
                return loss(D=D, bs=bs, mask=mask)

            optimizer.step(closure)
        with torch.no_grad():
            D *= mask
        return D, losses_inside

    num_iter_outer = 1000 
    num_iter_inner = 1000
    bs = 64
    D = D0.clone().requires_grad_()
    optimizer = opt.UniversalSGD([D], D=distance)
    times = [0.]
    main_losses = [full_loss(D).item()]
    total_iters = (num_iter_outer + num_iter_inner) // 2

    for j in tqdm(range(epochs)):
        time_start = time.time()
        D, _ = quadratic_optimization(D, optimizer, num_iter_outer, mask=main_mask)
        with torch.no_grad():
            order = projection_order1(D)
            mask = mask_from_order(order, main_mask)
            time_end = time.time()
            times.append(times[-1] + (time_end - time_start))
            loss_val = full_loss(D * mask).item()
            main_losses.append(loss_val)
        
        time_start = time.time()
        D, _ = quadratic_optimization(D, optimizer, num_iter_inner, mask=mask)
        with torch.no_grad():
            time_end = time.time()
            times.append(times[-1] + (time_end - time_start))
            loss_val = full_loss(D * mask).item()
            main_losses.append(loss_val)

    # Optionally save results
    with open(f'd={d}_n=5000_{graph_type}{degree}_{noise_type}.pickle', 'wb') as handle:
        pickle.dump((main_losses, times, optimal_test_loss, total_iters), handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run optimization on generated DAG.')
    
    parser.add_argument('--d', type=int, default=10, help='Number of nodes in the graph')
    parser.add_argument('--noise_type', type=str, default='gaussian_ev', help='Type of noise to use')
    parser.add_argument('--graph_type', type=str, default='ER', help='Type of graph to generate')
    parser.add_argument('--degree', type=int, default=2, help='Degree of the graph')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to run')
    parser.add_argument('--train_size',type=int, default=5000, help='Train size')
    parser.add_argument('--test_size',type=int, default=10000, help='Test size')
   
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    main(d=args.d, noise_type=args.noise_type, graph_type=args.graph_type, degree=args.degree, epochs=args.epochs, train_size=args.train_size, test_size=args.test_size, seed=args.seed)

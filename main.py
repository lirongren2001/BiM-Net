import argparse
import torch
import numpy as np
import random
from train import Trainer
from dataset import BrainGraphDataset

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_arguments():
    parser = argparse.ArgumentParser(description='Classification for Brain Graph')
    parser.add_argument('--k', type=int, default=5, help='number of nearest neighbors')
    parser.add_argument('--knn_lambda', type=float, default=0.1, help='weight for knn loss')
    parser.add_argument('--input_dim', type=int, default=240, help='fc_input_feature_dim')
    parser.add_argument('--num_nodes_wm', type=int, default=48, help='num_nodes')
    parser.add_argument('--num_nodes_gm', type=int, default=82, help='num_nodes')
    parser.add_argument('--datapath', type=str, default='', help='path of dataset')
    parser.add_argument('--repetitions', type=int, default=10, help='number of repetitions')
    parser.add_argument('--epochs', type=int, default=120, help='maximum number of epochs')
    parser.add_argument('--folds', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda devices')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden_dim')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--layer', type=int, default=1, help='the numbers of convolution layers') 
    parser.add_argument('--adv_lambda', type=float, default=0.01, help='adversarial loss weight')
    parser.add_argument('--dropout', type=float, default=0, help='dropout ratio') 
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--data_seed', type=int, default=[25, 50, 100, 125, 150, 175, 200, 225, 250, 275], help='data seed')  
    parser.add_argument('--savepath', type=str, default='ckpt/', help='path of model')
    parser.add_argument('--resultpath', type=str, default='results/test.txt', help='path of results')
    
    args = parser.parse_args()
    return args

def compute_variance(metrics_list):
    std_dev = {metric: np.std([metrics[metric] for metrics in metrics_list], ddof=1) for metric in metrics_list[0]}
    return std_dev

def format_metric_with_std(value, std):
    return f"{value*100:.2f}(±{std*100:.2f})"

def main():
    print(torch.cuda.current_device())
    args = parse_arguments()
    setup_seed(args.seed)
    
    dataset = BrainGraphDataset(
        data_path=args.datapath,
        num_nodes_wm=args.num_nodes_wm,
        num_nodes_gm=args.num_nodes_gm,
        k_fold=args.folds
    )
    
    final_results = []
    
    with open(args.resultpath, 'a+') as f:
        f.write(f"Configuration: epochs={args.epochs}, hidden_dim={args.hidden_dim}, batch_size={args.batch_size}, "
                f"weight_decay={args.weight_decay}, layer={args.layer}, adv_lambda={args.adv_lambda}, "
                f"dropout={args.dropout}, seed={args.seed}, lr={args.lr}, data_seed={args.data_seed}\n")
        f.write("Repetition\t" + "\t".join(["acc", "sensitivity", "specificity", "f1", "auc"]) + "\n")
    
    for repetition in range(args.repetitions):
        print(f"\n--- Repetition {repetition + 1} ---")
        
        fold_results = []
        for fold in range(args.folds):
            train_loader, val_loader, test_loader = dataset.kfold_split(
                batch_size=args.batch_size, 
                test_index=fold
            )
            
            trainer = Trainer(
                input_dim=args.input_dim,
                num_nodes_wm=args.num_nodes_wm,
                num_nodes_gm=args.num_nodes_gm,
                device=args.device,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                layers=args.layer,
                hidden_dim=args.hidden_dim,
                adv_lambda=args.adv_lambda,
                savepath=args.savepath
            )
            
            metrics, _ = trainer.train_and_evaluate(
                train_loader, 
                val_loader, 
                test_loader, 
                epochs=args.epochs
            )
            print(f"\nFold {fold + 1}, Result: {metrics}")
            fold_results.append(metrics)
        
        avg_metrics = {k: np.mean([r[k] for r in fold_results]) for k in fold_results[0].keys()}
        std_metrics = compute_variance(fold_results)
        print("\nAverage Metrics:", avg_metrics)
        
        with open(args.resultpath, 'a') as f:
            f.write(f"Repetition {repetition + 1}\t")
            metrics_str = "\t".join([format_metric_with_std(avg_metrics[k], std_metrics[k]) 
                                   for k in avg_metrics.keys()])
            f.write(metrics_str + "\n")
        
        final_results.append(avg_metrics)
    
    avg_final_results = {k: np.mean([r[k] for r in final_results]) for k in final_results[0].keys()}
    std_final_results = compute_variance(final_results)
    print("\nFinal Average Metrics:", avg_final_results)
    
    with open(args.resultpath, 'a') as f:
        f.write("\nFinal Average Metrics:\t")
        final_metrics_str = "\t".join([format_metric_with_std(avg_final_results[k], std_final_results[k]) 
                                     for k in avg_final_results.keys()])
        f.write(final_metrics_str + "\n")

if __name__ == "__main__":
    main()
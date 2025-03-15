import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms as T
import random
import estimators
from dataset import HAM10kDataset, CIFAR10Dataset
import argparse


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HAM10k", help="Dataset to use")
    parser.add_argument("--class-wise", default=False, action="store_true",
                        help="Whether to compute class-wise intrinsic dimensionality")
    parser.add_argument('--k1', default=10, type=int)
    parser.add_argument('--bsize', default=16, type=int, help='batch size')
    parser.add_argument('--n-workers', default=1, type=int)
    parser.add_argument('--eval-every-k', default=False, action="store_true",
                        help="Whether to evaluate every k<=k1")
    parser.add_argument('--anchor-samples', default=0, type=int,
                        help="0 for using all samples from the training set")
    parser.add_argument('--anchor-ratio', default=0, type=float,
                        help="0 for using all samples from the training set")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    print("Using dataset: {}".format(args.dataset))
    print("Using class-wise: {}".format(args.class_wise))
    print("Using k1: {}".format(args.k1))

    # Set random seed for reproducibility
    seed = 0
    set_seed(seed)

    if args.dataset == "HAM10k":
        data_dir = "./data/HAM10k"
        train_df = pd.read_csv(os.path.join(data_dir, "train_df.csv"))
        split = np.load(os.path.join(data_dir, "train_val_test_split.npy"), allow_pickle=True).item()
        train_ids = split["train"]
        train_df = train_df.loc[train_df['image'].isin(train_ids)]
        train_image_dir = os.path.join(data_dir, "preprocessed_train_images")
        transforms = T.Compose([
              T.Resize((224, 298)),
              T.ToTensor(),
              T.Normalize(
                (0.6523304, 0.62197226, 0.61544853),
                (0.11362693, 0.16195466, 0.16857147)
              )
        ])
        if args.class_wise:
            labels = np.unique(train_df["label"].values)
            for label in labels:
                label_df = train_df.loc[train_df["label"] == label]
                dataset = HAM10kDataset(label_df, train_image_dir, transform=transforms)
                dim, inv_mle_dim = estimators.mle_inverse_singlek(dataset, k1=args.k1, args=args)
                print(f"Label: {label}\tDim: {dim:.3f}\tInv MLE Dim: {inv_mle_dim:.3f}")
        else:
            dataset = HAM10kDataset(train_df, train_image_dir, transform=transforms)
            dim, inv_mle_dim = estimators.mle_inverse_singlek(dataset, k1=args.k1, args=args)
            print(f"Dim: {dim:.3f}\tInv MLE Dim: {inv_mle_dim:.3f}")

    elif args.dataset == "CIFAR10":
        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if args.class_wise:
            for label in range(10):
                dataset = CIFAR10Dataset(root="./data", train=True, transform=transforms, class_label=label)
                dim, inv_mle_dim = estimators.mle_inverse_singlek(dataset, k1=args.k1, args=args)
                print(f"Label: {label}\tDim: {dim:.3f}\tInv MLE Dim: {inv_mle_dim:.3f}")
        else:
            dataset = CIFAR10Dataset(root="./data", train=True, transform=transforms)
            dim, inv_mle_dim = estimators.mle_inverse_singlek(dataset, k1=args.k1, args=args)
            print(f"Dim: {dim:.3f}\tInv MLE Dim: {inv_mle_dim:.3f}")

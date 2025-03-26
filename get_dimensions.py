import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms as T
import random
import estimators
from dataset import HAM10kDataset, EmbeddingDataset, CIFAR10Dataset, FitzpatrickDataset
import argparse
from sklearn.model_selection import train_test_split


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HAM10k", help="Dataset to use")
    parser.add_argument(
        "--class-wise",
        default=False,
        action="store_true",
        help="Whether to compute class-wise intrinsic dimensionality",
    )
    parser.add_argument(
        "--embedding-space",
        default=False,
        action="store_true",
        help="Whether to compute intrinsic dimensionality in the embedding space",
    )
    parser.add_argument(
        "--embedding-path", default=None, type=str, help="Path to the embeddings"
    )
    parser.add_argument("--k1", default=10, type=int)
    parser.add_argument("--bsize", default=16, type=int, help="batch size")
    parser.add_argument("--n-workers", default=1, type=int)
    parser.add_argument(
        "--eval-every-k",
        default=False,
        action="store_true",
        help="Whether to evaluate every k<=k1",
    )
    parser.add_argument(
        "--anchor-samples",
        default=0,
        type=int,
        help="0 for using all samples from the training set",
    )
    parser.add_argument(
        "--anchor-ratio",
        default=0,
        type=float,
        help="0 for using all samples from the training set",
    )
    parser.add_argument(
        "--class-level",
        default="low",
        type=str,
        help="Class level (e.g., low, mid, high) for Fitzpatrick17k",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    print("Using dataset: {}".format(args.dataset))
    print("Using class-wise: {}".format(args.class_wise))
    print("Using k1: {}".format(args.k1))

    # Set random seed for reproducibility
    seed = 0
    set_seed(seed)

    dims = {args.k1: []}
    save_path = f"./estimates/{args.dataset}/"
    if args.embedding_space:
        save_path += "embedding_space/"
        save_path += f"{args.embedding_path.split('/')[-1].split('.')[0]}/"
    else:
        save_path += "image_space/"
    if args.class_wise:
        save_path += "class_wise/"
    else:
        save_path += "all_classes/"
    save_path += f"{args.k1}"
    os.makedirs(save_path, exist_ok=True)
    save_path += "/estimates.npy"

    if args.embedding_space:
        assert (
            args.embedding_path is not None
        ), "Please provide the path to the embeddings"
        print("Using embeddings from: {}".format(args.embedding_path))
        train_df = np.load(args.embedding_path, allow_pickle=True).item()["train"]
        # TODO: should we only be conisdering training data when computing inv_dim?
        if args.class_wise:
            labels = np.unique(train_df["label"].values)
            for label in labels:
                label_df = train_df.loc[train_df["label"] == label]
                dataset = EmbeddingDataset(label_df)
                dim, inv_mle_dim = estimators.mle_inverse_singlek(
                    dataset, k1=args.k1, args=args
                )
                dims[args.k1].append(inv_mle_dim)
        else:
            dataset = EmbeddingDataset(train_df)
            dim, inv_mle_dim = estimators.mle_inverse_singlek(
                dataset, k1=args.k1, args=args
            )
            dims[args.k1].append(inv_mle_dim)

    else:
        if args.dataset == "HAM10k":
            data_dir = "./data/HAM10k"
            train_df = pd.read_csv(os.path.join(data_dir, "train_df.csv"))
            split = np.load(
                os.path.join(data_dir, "train_val_test_split.npy"), allow_pickle=True
            ).item()
            train_ids = split["train"]
            train_df = train_df.loc[train_df["image"].isin(train_ids)]
            train_image_dir = os.path.join(data_dir, "preprocessed_train_images")
            transforms = T.Compose(
                [
                    T.Resize((224, 298)),
                    T.ToTensor(),
                    T.Normalize(
                        (0.6523304, 0.62197226, 0.61544853),
                        (0.11362693, 0.16195466, 0.16857147),
                    ),
                ]
            )
            if args.class_wise:
                labels = np.unique(train_df["label"].values)
                for label in labels:
                    label_df = train_df.loc[train_df["label"] == label]
                    dataset = HAM10kDataset(
                        label_df, train_image_dir, transform=transforms
                    )
                    dim, inv_mle_dim = estimators.mle_inverse_singlek(
                        dataset, k1=args.k1, args=args
                    )
                    dims[args.k1].append(inv_mle_dim)
            else:
                dataset = HAM10kDataset(train_df, train_image_dir, transform=transforms)
                dim, inv_mle_dim = estimators.mle_inverse_singlek(
                    dataset, k1=args.k1, args=args
                )
                dims[args.k1].append(inv_mle_dim)

        elif args.dataset == "CIFAR10":
            transforms = T.Compose(
                [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
            if args.class_wise:
                for label in range(10):
                    dataset = CIFAR10Dataset(
                        root="./data",
                        train=True,
                        transform=transforms,
                        class_label=label,
                    )
                    dim, inv_mle_dim = estimators.mle_inverse_singlek(
                        dataset, k1=args.k1, args=args
                    )
                    dims[args.k1].append(inv_mle_dim)
            else:
                dataset = CIFAR10Dataset(
                    root="./data", train=True, transform=transforms
                )
                dim, inv_mle_dim = estimators.mle_inverse_singlek(
                    dataset, k1=args.k1, args=args
                )
                dims[args.k1].append(inv_mle_dim)

        elif args.dataset == "Fitzpatrick17k":
            df = pd.read_csv("./data/fitzpatrick17k/fitzpatrick17k.csv")
            imgs_to_drop = [
                "2ea034fc482e9bd21a5dfa2506cc5d6b",
                "bccab73c32aba48c90602de9bb458272",
                "e69165b3455bb3a5a8b33a0f6fd8a1d3",
            ]  # erroneous images that are not dermatology related
            df = df[~df["md5hash"].isin(imgs_to_drop)]

            df["low"] = df["label"].astype("category").cat.codes
            df["mid"] = df["nine_partition_label"].astype("category").cat.codes
            df["high"] = df["three_partition_label"].astype("category").cat.codes

            train_df, _ = train_test_split(
                df,
                test_size=0.2,
                random_state=seed,
                shuffle=True,
                stratify=df[args.class_level],
            )
            train_df, _ = train_test_split(
                train_df,
                test_size=0.1,
                random_state=seed,
                shuffle=True,
                stratify=train_df[args.class_level],
            )
            transforms = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            if args.class_wise:
                labels = np.unique(train_df[args.class_level].values)
                for label in labels:
                    label_df = train_df.loc[train_df[args.class_level] == label]
                    dataset = FitzpatrickDataset(
                        label_df,
                        args.class_level,
                        "./data/fitzpatrick17k/preprocessed_images",
                        transform=transforms,
                    )
                    dim, inv_mle_dim = estimators.mle_inverse_singlek(
                        dataset, k1=args.k1, args=args
                    )
                    dims[args.k1].append(inv_mle_dim)
            else:
                dataset = FitzpatrickDataset(
                    train_df,
                    args.class_level,
                    "./data/fitzpatrick17k/preprocessed_images",
                    transform=transforms,
                )
                dim, inv_mle_dim = estimators.mle_inverse_singlek(
                    dataset, k1=args.k1, args=args
                )
                dims[args.k1].append(inv_mle_dim)

    np.save(save_path, dims)

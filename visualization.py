#!/usr/bin/env python3
"""Visualize the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import utils
from model.net import Net
import model.data_loader as d_l


def args_parser():
    """Parse commadn line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default='../datasets/',
                        help="Directory containing the dataset")
    parser.add_argument('--model_dir',
                        default='experiments/base_model',
                        help="Directory containing params.json")
    parser.add_argument('--restore_file',
                        default='best',
                        help="name of the file in --model_dir containing weights to load")
    return parser.parse_args()


def visualize(model, dataloader, params, writer, num_proj=100):
    """Evaluate the model visualize the results.
    Args:
        model: (torch.nn.Module) the neural network
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        params: (Params) hyperparameters
        writer: (SummaryWriter) Summary writer for tensorboard
        num_proj: (int) Number of images to project
    """
    # put model in evaluation mode
    model.eval()
    class_probs = []
    embeddings = []
    inputs = []
    labels = []

    with torch.no_grad():
        for _, input_data in enumerate(dataloader):
            inp_data, label = input_data
            # move data to GPU if possible
            if params.cuda:
                inp_data = inp_data.to(params.device)
                label = label.to(params.device)

            # compute model output
            embed, output = model(inp_data)

            # move to cpu
            for x in [output, embed, inp_data, label]:
                x = x.cpu()

            class_probs.append(output)
            embeddings.append(embed)
            inputs.append(inp_data)
            labels.append(label)


    logging.info("- done.")

    class_probs = torch.exp(torch.cat(class_probs))
    _, class_preds = torch.max(class_probs, 1)
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)
    inputs = torch.cat(inputs)

    # Add PR curve to tensorboard
    logging.info("Add Precision-Recall in tensorboard...")
    for i in range(params.num_classes):
        tensorboard_probs = class_probs[:, i]
        tensorboard_preds = class_preds == i

        writer.add_pr_curve(str(i),
                            tensorboard_preds,
                            tensorboard_probs)

    # Add random samples to the projector on tensorboard
    logging.info("Add Projections in tensorboard...")

    perm = np.random.randint(0, len(embeddings), num_proj)
    labels = labels[perm]
    class_labels = labels.tolist()
    writer.add_embedding(mat=embeddings[perm, ...],
                         metadata=class_labels,
                         label_img=inputs[perm, ...])


def main():
    """Main function
    """
    # Load the parameters
    args = args_parser()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json config file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Create summary writer for use with tensorboard
    writer = SummaryWriter(os.path.join(args.model_dir, 'runs', 'visualize'))

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
        params.device = "cuda:0"
    else:
        params.device = "cpu"

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'visualize.log'))

    logging.info("Loading the dataset...")

    # fetch dataloaders
    dataloaders = d_l.get_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = Net(params)
    if params.cuda:
        model = model.to(params.device)

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Visuzlize
    visualize(model, test_dl, params, writer)

    writer.close()


if __name__ == '__main__':
    main()

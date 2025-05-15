import os
import sys
import logging
import argparse
sys.path.append('C:/Users/farid/PycharmProjects/FaultDetectionProject/OpenStack/')
import torch
torch.manual_seed(0)
from Trainer.TrainerOpenStack import train

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # S3 bucket and prefix for remote mode
    parser.add_argument('--bucket', type=str, default=None,
                        help='S3 bucket name for remote data (required if not local)')
    parser.add_argument('--prefix', type=str, default='',
                        help='S3 prefix/folder for remote data')

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--window-size', type=int, default=10, metavar='N',
                        help='length of training window (default: 10)')
    parser.add_argument('--input-size', type=int, default=1, metavar='N',
                        help='model input size (default: 1)')
    parser.add_argument('--hidden-size', type=int, default=64, metavar='N',
                        help='hidden layer size (default: 64)')
    parser.add_argument('--num-layers', type=int, default=2, metavar='N',
                        help='number of model\'s layer (default: 2)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--num-classes', type=int, metavar='N',
                        help='the number of model\'s output, must same as pattern size!')
    parser.add_argument('--num-candidates', type=int, metavar='N',
                        help='the number of predictors sequences as correct predict.')

    # Pass container environment
    parser.add_argument('--hosts', type=list, default=['127.0.0.1'],
                        help='args for SageMaker distributed training.')
    parser.add_argument('--current-host', type=str, default='127.0.0.1',
                        help='args for SageMaker distributed training.')
    parser.add_argument('--model-dir', type=str, default='./model/',
                        help='the place where to store the model parameter.')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='the place where to store the training data.')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpu to train')

    # Local mode (set default to True for most users)
    parser.add_argument('--local', type=bool, default=True,
                        help='local training model (default: True).')

    args = parser.parse_args()

    # Ensure model directory exists
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)

    # If not local, require bucket
    if not args.local and not args.bucket:
        raise ValueError("S3 bucket name must be provided for non-local mode. Use --bucket your-bucket-name")

    train(args)
    logger.debug('Finished Training')

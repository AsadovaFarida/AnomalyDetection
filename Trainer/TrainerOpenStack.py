import os
import sys
import json
import logging
import argparse
import os
import torch

import boto3
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Use environment variables or command-line arguments for S3 config
BUCKET = os.environ.get('S3_BUCKET', None)
PREFIX = os.environ.get('S3_PREFIX', '')


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(input.device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(input.device)
        out, _ = self.lstm(input, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class Generate():
    def __init__(self):
        self.init_obj = None

    def init_line(self, local, name, bucket=None, prefix=''):
        if local:
            f = open(name, 'r')
            self.init_obj = f
            line = self.init_obj.readline()
        else:
            if not bucket:
                raise ValueError("S3 bucket name must be provided for non-local mode.")
            client = boto3.client('s3')
            obj = client.get_object(Bucket=bucket, Key=prefix + name)
            self.init_obj = obj['Body']
            line = self.init_obj.readline()
            line = line.decode().rstrip()
        return line

    def readline(self, local):
        if local:
            line = self.init_obj.readline()
        else:
            line = self.init_obj.readline()
            line = line.decode().rstrip()
        return line

    def generate(self, name, window_size, local, bucket=None, prefix=''):
        num_sessions = 0
        inputs = []
        outputs = []

        line = self.init_line(local, name, bucket, prefix)
        while line:
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i+window_size])
                outputs.append(line[i+window_size])
            line = self.readline(local)
            num_sessions += 1
        logger.info('Number of session({}): {}'.format(name, num_sessions))
        logger.info('Number of seqs({}): {}'.format(name, len(inputs)))
        if outputs:
            max_label = max(outputs)
            logger.info(f"Max label in dataset: {max_label}")
        else:
            max_label = 0
        dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs, dtype=torch.long))
        return dataset, max_label

def _get_train_data_loader(batch_size, is_distributed, window_size, local, bucket=None, prefix='', **kwargs):
    logger.info("Get train data loader")
    _generate = Generate()
    seq_dataset, max_label = _generate.generate(name='train', window_size=window_size, local=local, bucket=bucket, prefix=prefix)
    train_sampler = torch.utils.data.distributed.DistributedSampler(seq_dataset) if is_distributed else None
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=train_sampler is None,
                            sampler=train_sampler, **kwargs)
    return dataloader, max_label

# ... rest of your code remains unchanged until train() ...
# ...existing code...
def save_model(model, model_dir, args):
        """Save the PyTorch model to the specified directory."""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, "model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'args': vars(args)
        }, model_path)
        print(f"Model saved to {os.path.join(args.model_dir, 'model.pth')}")
        
def train(args):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    if args.num_gpus > 0 and not torch.cuda.is_available():
        logger.warning("No CUDA available, setting num_gpus to 0 (num_gpus = {}).".format(args.num_gpus))
        args.num_gpus = 0
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus requested - {}, available - {}.".format(args.num_gpus, torch.cuda.device_count()))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        logger.info('Initialize the distributed environment')
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info('Initialized the distributed environment:\'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))

    torch.manual_seed(args.seed)
    if use_cuda:
        logger.info('Use CUDA')
        torch.cuda.manual_seed(args.seed)

    # Pass bucket and prefix to data loader
    train_loader, max_label = _get_train_data_loader(
        args.batch_size, is_distributed, args.window_size, args.local,
        bucket=args.bucket, prefix=args.prefix, **kwargs
    )

    # Dynamically set num_classes if not provided or too small
    if args.num_classes is None or args.num_classes <= max_label:
        args.num_classes = max_label + 1
        logger.info(f"Setting num_classes to {args.num_classes} based on max label.")

    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    model = Model(args.input_size, args.hidden_size, args.num_layers, args.num_classes).to(device)
    if is_distributed:
        if use_cuda:
            logger.info('multi-machine multi-gpu case')
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            logger.info('single-machine multi-gpu case or single-machine or multi-machine cpu case')
            model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for seq, label in train_loader:
            seq = seq.clone().detach().view(seq.size(0), args.window_size, args.input_size).to(device)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, label.to(device))
            loss.backward()
            if is_distributed and not use_cuda:
                _average_gradients(model)
            optimizer.step()
            train_loss += loss.item()
        logger.debug('Epoch [{}/{}], Train_loss: {}'.format(
            epoch, args.epochs, round(train_loss/len(train_loader.dataset), 4)
        ))
    logger.debug('Finished Training')
    save_model(model, args.model_dir, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Set a default for num-classes (can be overridden by data)
    parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                        help='the number of model\'s output, must same as pattern size!')
    parser.add_argument('--num-candidates', type=int, metavar='N',
                        help='the number of predictors sequences as correct predict.')

    # Container environment (use safe defaults for local runs)
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS', '["localhost"]')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST', 'localhost'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data'))
    parser.add_argument('--num-gpus', type=int, default=int(os.environ.get('SM_NUM_GPUS', 0)))

    # S3 bucket and prefix for remote mode
    parser.add_argument('--bucket', type=str, default=os.environ.get('S3_BUCKET', None),
                        help='S3 bucket name for remote data (required if not local)')
    parser.add_argument('--prefix', type=str, default=os.environ.get('S3_PREFIX', ''),
                        help='S3 prefix/folder for remote data')

    # Local mode
    parser.add_argument('--local', type=bool, default=True,
                        help='local training model (default: True).')

    args = parser.parse_args()

    # Print model parameters for debugging
    print(f"input_size={args.input_size}, hidden_size={args.hidden_size}, num_layers={args.num_layers}, num_classes={args.num_classes}")

    train(args)
# ...existing code...

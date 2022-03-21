import logging
import sys
from pathlib import Path

import pandas as pd
from torchvision import transforms
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

from SudokuDataset import SudokuDataset
from data_handle import CarvanaDataset, SimDataset
from dice_score import dice_loss
from evaluate import evaluate
from unet_module import UNET


N_CLASSES = 1
dir_checkpoint = Path('./checkpoints/')

def training(model, optimizer, criterion, trainloader, writer):
    running_loss = 0.0
    for epoch in range(1):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:  # every 1000 mini-batches...

                # ...log the running loss
                writer.add_scalar('training loss',
                                  running_loss / 1000,
                                  epoch * len(trainloader) + i)
                running_loss = 0.0
    print('Finished Training')
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def test():
    x = torch.randn((300, 3, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape


def train_net(net, dataset, epochs, batch_size, learning_rate, device, writer,
              save_checkpoint=True, val_percent=0.1, amp=True):
    pass

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch[0]
                true_masks = batch[1]

                # assert images.shape[1] == net.in_channels, \
                #     f'Network has been defined with {net.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float16)
                true_masks = true_masks.to(device=device, dtype=torch.float16)

                with torch.cuda.amp.autocast(enabled=amp, dtype=torch.float16):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks)
                           # + dice_loss(F.softmax(masks_pred, dim=1).float(),
                           #             F.one_hot(true_masks, N_CLASSES).permute(0, 3, 1, 2).float(),
                           #             multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                writer.add_scalar('training loss',
                                  loss.item(),
                                  global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (4 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        # for tag, value in net.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))

                        writer.add_scalar('validation_loss',
                                          val_score,
                                          global_step)


        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    writer = SummaryWriter('runs/unet_testing_1')

    args = objectview({
        'epochs': 10,
        'batch_size': 128,
        'lr': 1e-5,
        'val': 1,
        'amp': True,
    })

    model = UNET(in_channels=10, out_channels=9)
    data = pd.read_csv('c:/Users/petro/Documents/sudoku-3m.csv', sep=',')



    dataset = SudokuDataset(data)

    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
    ])

    # dataset = SimDataset(10**4, 45, transform=trans)
    model.to(device=device)
    try:
        train_net(net=model,
                  dataset=dataset,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  writer=writer,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)


    test()

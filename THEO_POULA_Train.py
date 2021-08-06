## package load
#cOMMENT TO PUSH AS A TEST
import numpy as np
import time
import torch
import os
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from THEO_POULA_Model import VGG, LSTMModel
import pickle as pkl
import argparse
from THEO_POULA_Optim import THEOPOULA

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser(description = 'pytorch CIFAR10')
parser.add_argument('--trial', default='trial1', type=str)
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--model', default='VGG11', type=str)
parser.add_argument('--num_epoch', default=200, type=int, dest='num_epoch')
parser.add_argument('--optimizer_name', default='THEOPOULA', type=str)
parser.add_argument('--eta', default='0', type=float)
parser.add_argument('--beta', default='1e14', type=float)
parser.add_argument('--r', default=5, type=int)
parser.add_argument('--eps', default=1e-4, type=float)
parser.add_argument('--act_fn', default='silu', type=str)

parser.add_argument('--log_dir', default='./log/', type=str)
parser.add_argument('--ckpt_dir', default='./ckpt/', type=str)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_time = time.time()

best_loss = 999
state = []
start_epoch = 1
trial = args.trial
batch_size = args.batch_size
num_epoch = args.num_epoch
optimizer_name = args.optimizer_name
act_fn = args.act_fn
model = args.model
lr = args.lr
eta = args.eta
beta = args.beta
r = args.r
eps = args.eps

## Preparing data and dataloader
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader= torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle = False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


num_data = len(trainloader.dataset)
num_batch = np.ceil(num_data / batch_size)


torch.manual_seed(1111)
## Model

print('==> Building model.. on {%s}'%device)
net = VGG(args.model, act_fn)
net.to(device)
criterion = nn.MSELoss()

print('==> Setting optimizer.. {%s}'%optimizer_name)
if optimizer_name == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=lr)
elif optimizer_name == 'RMSProp':
    optimizer = optim.RMSprop(net.parameters(), lr=lr)
elif optimizer_name == 'ADAM':
    optimizer = optim.Adam(net.parameters(), lr=lr)
elif optimizer_name == 'AMSGrad':
    optimizer = optim.Adam(net.parameters(), lr=lr, amsgrad=True)
elif optimizer_name == 'THEOPOULA':
    optimizer = THEOPOULA(net.parameters(), lr=lr, eta=eta, beta=args.beta, r=r, eps=eps)



if optimizer_name == 'THEOPOULA':
    experiment_name = '%s_%s_bs{%d}_lr{%.1e}_epoch{%d}_eta{%.1e}_beta{%.1e}_r{%d}_eps{%.1e}' \
                      %(optimizer_name, model, batch_size, lr, num_epoch, eta, beta, r, eps)
else:
    experiment_name = '%s_%s_bs{%d}_lr{%.1e}_eps{%.1e}_epoch{%d}'%(optimizer_name, model, batch_size, lr, eps, num_epoch)


log_dir = args.log_dir + experiment_name
ckpt_dir = args.ckpt_dir + experiment_name

fn_pred = lambda output: torch.softmax(output, dim=1)
fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()

writer = SummaryWriter(log_dir=log_dir)

## Training

history = {'training_loss': [],
           'test_loss': [],
           'training_acc': [],
           'test_acc': [],
           }


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = []
    acc_arr = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        targets_ohe = F.one_hot(targets, num_classes=10).float()
        inputs, targets, targets_ohe = inputs.to(device), targets.to(device), targets_ohe.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        pred = fn_pred(outputs)
        loss = criterion(outputs, targets_ohe)
        acc = fn_acc(pred, targets)

        loss.backward()
        optimizer.step()

        train_loss += [loss.item()]
        acc_arr += [acc.item()]

        if batch_idx%200 == 0:
            print('TRAIN: EPOCH %04d/%04d | BATCH %04d/%04d | LOSS: %.4f |  ACC %.4f' %
              (epoch, args.num_epoch, batch_idx, num_batch, train_loss[-1], acc_arr[-1]))
    print('TRAIN: EPOCH %04d/%04d | LOSS: %.4f |  ACC %.4f' %
          (epoch, args.num_epoch, np.mean(train_loss), np.mean(acc_arr)))
    writer.add_scalar('Training loss', np.mean(train_loss), epoch)
    writer.add_scalar('Training accuracy', np.mean(acc_arr), epoch)

    history['training_loss'].append(np.mean(train_loss))
    history['training_acc'].append(np.mean(acc_arr))


def test(epoch):
    global best_loss, state
    net.eval()
    test_loss = []
    acc_arr = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets_ohe = F.one_hot(targets, num_classes=10).float()
            inputs, targets, targets_ohe = inputs.to(device), targets.to(device), targets_ohe.to(device)
            outputs = net(inputs)
            pred = fn_pred(outputs)
            loss = criterion(outputs, targets_ohe)
            acc = fn_acc(pred, targets)

            test_loss += [loss.item()]
            acc_arr += [acc.item()]

        print('TEST:  LOSS: %.4f |  ACC %.4f' %
                  (np.mean(test_loss),  np.mean(acc_arr)))

    if np.mean(test_loss) < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': np.mean(test_loss),
            'epoch': epoch,
            'optim': optimizer.state_dict()
        }
        best_loss = np.mean(test_loss)

    writer.add_scalar('Test loss', np.mean(test_loss), epoch)
    writer.add_scalar('Test accuracy', np.mean(acc_arr), epoch)

    history['test_loss'].append(np.mean(test_loss))
    history['test_acc'].append(np.mean(acc_arr))


for epoch in range(start_epoch, start_epoch + num_epoch):
    train(epoch)
    test(epoch)


elapsed_time = time.time() - start_time
print(elapsed_time)


plt.figure(1)
plt.plot(range(1, num_epoch+1), history['training_loss'], label='train')
plt.plot(range(1, num_epoch+1), history['test_loss'], label='test')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
pkl.dump(history, open(log_dir+'/history.pkl', 'wb'))


if not os.path.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)
torch.save(state, './%s/%s.pth' % (ckpt_dir, experiment_name))


#plt.show()
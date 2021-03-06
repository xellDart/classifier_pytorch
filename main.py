import argparse
from dataprocess.dataset import Dataset
from models import ShuffleNetV2
from models import MobileNet2
from models import MobileNetV3, BetterShuffleNet, SENet18, mnasnet
import torch as t
from torch.utils import data
import math
import torch.optim.lr_scheduler as lr_scheduler
import copy
from torch.cuda import amp
from utils.utils import CrossEntropyLabelSmooth


ap = argparse.ArgumentParser()
ap.add_argument("-gpu", "--use_gpu", type=int, default=1,
                help="use gpu or not")
ap.add_argument("-bs", "--batchsize", type=int, default=128,
                help="the batch size of input")
ap.add_argument("-t", "--train", type=int, default=1,
                help="choose training or valdating")
ap.add_argument("-pre", "--pretrained", type=str, default="None",
                help="select a pretrained model")
ap.add_argument("-e", "--epochs", type=int, default=300,
                help="epochs of training")
ap.add_argument("-path", "--datapath", type=str, default="./dataset/train",
                help="path of training dataset")
# the arguments for model
ap.add_argument("-m", "--model", type=str, default="ShuffleNet2",
                help="the type of model")
ap.add_argument("-c", "--classes", type=int, default=2,
                help="the number of classes of dataset")
ap.add_argument("-s", "--inputsize", type=int, default=224,
                help="the size of image")
ap.add_argument("-nt", "--nettype", type=int, default=1,
                help="type of network")


def train_model(model, dataloaders, optimizer, loss_fn, scheduler, num_epochs=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.
    val_acc_history = []

    for epoch in range(num_epochs):

        print("Epoach: {}".format(epoch))

        for phase in ["train", "val"]:
            running_loss = 0.
            running_corrects = 0.

            if phase == "train":
                model.train()
            else:
                model.eval()

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                with t.autograd.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    with amp.autocast(enabled=False):  # stability issues when enabled
                        loss = loss_fn(outputs, labels)

                preds = outputs.argmax(dim=1)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += t.sum(preds.view(-1)
                                          == labels.view(-1)).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print("Phase {} loss: {}, acc: {}".format(
                phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def test_model(model, dataloader, loss_fn):
    import time
    model.eval()
    running_loss = 0.
    running_corrects = 0.
    records = []

    for inputs, labels in dataloader:
        img_len = len(inputs)
        inputs, labels = inputs.to(device), labels.to(device)

        #
        start = time.time()
        outputs = model(inputs)
        #
        end = time.time()
        fps = img_len/(end-start)
        records.append(fps)
        loss = loss_fn(outputs, labels)
        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += t.sum(preds.view(-1) == labels.view(-1)).item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)

    print("On val dataset loss: {}, acc: {}".format(epoch_loss, epoch_acc))
    import numpy as np
    print("{} FPS".format(np.mean(records)))
    return model


def convert_torch(model, input_size):
    device = t.device('cpu')
    model.to(device)
    model.eval()

    input = t.rand(1, 3, input_size, input_size)
    traced_script_module = t.jit.trace(model, input)
    t.jit.save(traced_script_module, 'model_out.pth')

if __name__ == '__main__':
    args = vars(ap.parse_args())
    path = args["datapath"]
    train_sign = args["train"]
    epochs = args["epochs"]
    num_classes = args["classes"]
    input_size = args["inputsize"]
    net_type = args["nettype"]
    model_type = args["model"]

    batchsize = args["batchsize"]
    dataloader = {}
    if train_sign:
        train_dataset = Dataset(
            "./dataset/train", train=True, input_size=input_size)

        train_loader = data.DataLoader(train_dataset,
                                       batch_size=batchsize,
                                       num_workers=4,
                                       shuffle=True)
        dataloader["train"] = train_loader

    val_dataset = Dataset("./dataset/train", train=False,
                          test=False, input_size=input_size)
    val_loader = data.DataLoader(val_dataset,
                                 batch_size=batchsize,
                                 num_workers=4,
                                 shuffle=True)
    dataloader["val"] = val_loader

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model_path = args["pretrained"]

    if model_type == "ShuffleNet2":
        model = ShuffleNetV2(num_classes=num_classes)
    elif model_type == "MobileNet2":
        model = MobileNet2(num_classes, input_size, net_type)
    elif model_type == "MobileNetV3_Large":
        model = MobileNetV3(subtype='large', n_class=num_classes)
    elif model_type == "MobileNetV3_Small":
        model = MobileNetV3(subtype='mobilenet_v3_small', n_class=num_classes)
    elif model_type == "BetterShuffleNet":
        model = BetterShuffleNet(num_classes)
    elif model_type == "SENet18":
        model = SENet18(num_classes)
    elif model_type == "MnasNet":
        model = mnasnet.MnasNet(num_classes)
    else:
        print("We don't implement the model, please choose ShuffleNet2 or MobileNet2")
    if model_path != "None":
        model.load_state_dict(t.load(model_path))
    model.to(device)

    lr0 = 0.0001 * batchsize  # intial lr
    lrf = 0.01  # final lr (fraction of lr0)

    loss_fn = CrossEntropyLabelSmooth(num_classes, 0.1)
    #optimizer = t.optim.AdamW(model.parameters(), lr=lr0 / 10)
    optimizer = t.optim.SGD(model.parameters(), lr=lr0, momentum=0.9, nesterov=True)

    def lf(x): return ((1 + math.cos(x * math.pi / epochs)) / 2) * \
        (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    if train_sign:
        #model, dataloaders, optimizer, loss_fn, scheduler, num_epochs=5
        model, val_logs = train_model(
            model, dataloader, optimizer, loss_fn, scheduler,  epochs)
        # store the model
        import time
        pkl_path = "./save/" + model_type + str(int(time.time()))+'.pkl'
        t.save(model.state_dict(), pkl_path)
        print("Model saved to", pkl_path)
    else:
        model = test_model(model, dataloader['val'], loss_fn)
        convert_torch(model, input_size=input_size)

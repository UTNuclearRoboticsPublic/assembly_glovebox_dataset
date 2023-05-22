import torch
from tqdm import tqdm
from utils import *

def train_fn(loader, model, optimizer, loss_fn):
    model.train()

    train_loss = 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)
        model = model.to(DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data)

            # instead, adjust the FastSCNN architecture to only use first index
            if architecture == "FastSCNN":
                prediction = predictions[0]

            loss = loss_fn(prediction, targets)

            train_loss += loss.item()

            probabilities = torch.softmax(prediction, dim=1)
            predictions = torch.argmax(predictions, dim=1).detach()
            final_predictions = predictions.to(device=DEVICE)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = train_loss / len(loader)
    
    return final_loss


            


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
    # parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    # parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    # parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    # parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    # parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')

    nets = []
    optimizers = []
    for _ in range(NUM_NETS):
        if architecture == "UNET_DROPOUT":
            net = UNET_Dropout(in_channels=3, out_channels=3, droprate = 0.5)
        elif architecture == "UNET":
            net = UNET(in_channels=3, out_channels=3)
        elif architecture == "FASTSCNN":
            net = FastSCNN(in_channels=3, out_channels=3)
        nets.append(net)
        optimizers.append(torch.optim.Adam(net.parameters(), lr=LEARNING_RATE))
    
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader, val_loader = get_loaders()

    # load the model here

    # create example writer here and set up the metrics

    for i, net in enumerate(nets):
        optimizer = optimizers[i]
        model = nets[i].to(device)
        for epoch in range(NUM_EPOCHS):
            if LOAD_MODEL is not True:
                loss = train_fn(train_loader, model, optimizer, loss_fn)

if __name__ == "__main__":
    main()
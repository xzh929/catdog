from net import CNN_net
from data import dataset
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch
from torch.nn.functional import one_hot
from torch.utils.tensorboard import SummaryWriter


DEVICE = "cuda"

class Train:
    def __init__(self,root):
        self.train_data = dataset(root, is_train=True)
        self.train_loader = DataLoader(self.train_data,batch_size=100,shuffle=True)

        self.test_data = dataset(root,is_train=False)
        self.test_loader = DataLoader(self.test_data,batch_size=100,shuffle=True)

        self.summary = SummaryWriter()

        self.loss_fun = nn.BCEWithLogitsLoss()

        self.net = CNN_net().to(DEVICE)
        self.opt = optim.Adam(self.net.parameters())

    def __call__(self):
        for epoch in range(1000):
            sum_train_loss = 0.
            for i,(imgs, tags) in enumerate(self.train_loader):
                self.net.train()
                imgs = imgs.to(DEVICE)
                tags = tags.to(DEVICE)
                train_out = self.net(imgs)
                tags = one_hot(tags, 2)
                tags = tags.to(torch.float32)
                train_loss = self.loss_fun(train_out,tags)

                self.opt.zero_grad()
                train_loss.backward()
                self.opt.step()

                sum_train_loss += train_loss.item()
            avg_train_loss = sum_train_loss/len(self.train_loader)
            print("train_loss:",avg_train_loss)

            sum_score = 0.
            sum_test_loss = 0.
            for i, (imgs, tags) in enumerate(self.test_loader):
                self.net.eval()
                imgs = imgs.to(DEVICE)
                tags = tags.to(DEVICE)
                tags = one_hot(tags, 2)
                tags = tags.to(torch.float32)
                test_out = self.net(imgs)
                test_loss = self.loss_fun(test_out, tags)

                pre = torch.argmax(test_out,dim=1)
                tags = torch.argmax(tags,dim=1)
                score = torch.mean(torch.eq(pre,tags).float())
                sum_test_loss += test_loss.item()
                sum_score += score.item()
            avg_test_loss = sum_test_loss/len(self.test_loader)
            avg_score = sum_score/len(self.test_loader)
            self.summary.add_scalars("loss",{"train_loss":avg_train_loss,"test_loss":avg_test_loss},epoch)
            self.summary.add_scalar("score",avg_score,epoch)
            torch.save(self.net.state_dict(),f"checkpoint/{epoch}.t")
            print("epoch:",epoch,"test_loss:",avg_test_loss,"score:",avg_score)

if __name__ == '__main__':
    train = Train("D:\data\cat_dog")
    train()
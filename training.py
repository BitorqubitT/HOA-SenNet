import torch as tc 
import torch.nn as nn  
import numpy as np
from tqdm import tqdm
import cv2
from torch.cuda.amp import autocast
import albumentations as A
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from glob import glob
import helper
import ssl
from dotenv import load_dotenv

ssl._create_default_https_context = ssl._create_unverified_context
tc.cuda.is_available()
# Diceloss smooth factor to 0.1 (what does it do?)
# Save the params
# Bigger image size + lower precision maybe?

class CFG:
    target_size = 1
    model_name = 'Unet'
    backbone = 'se_resnext50_32x4d'

    in_chans = 1
    image_size = 1024
    input_size = 1024

    train_batch_size = 1
    valid_batch_size = train_batch_size * 2

    epochs = 30
    lr = 8e-5
    weight_decay = 1e-2
    chopping_percentile=1e-3
    valid_id = 1

    train_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightness(limit=0.1, p=0.7), # p = 0.7
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                A.MedianBlur(blur_limit=3),
                ], p=0.4),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=1., p=1.0)
        ],p=0.2),
        A.ShiftScaleRotate(p=0.7, scale_limit=0.5, shift_limit=0.2, rotate_limit=30),
        A.CoarseDropout(max_holes=1, max_height=0.25, max_width=0.25),
        A.RandomGamma(p=0.8), # check
        ToTensorV2(transpose_mask=True)
    ])

    valid_aug = A.Compose(ToTensorV2(transpose_mask=True))

# model
class CustomModel(nn.Module):
    def __init__(self, CFG, weight=None):
        super().__init__()
        self.model = smp.FPN(  #FPN Unet
            encoder_name=CFG.backbone, 
            encoder_weights=weight,
            in_channels=CFG.in_chans,
            classes=CFG.target_size,
            activation=None,
        )

    def forward(self, image):
        output = self.model(image)
        # output = output.squeeze(-1)
        return output[:,0]#.sigmoid()

def build_model(weight="imagenet"):
    load_dotenv()

    print(f'model_name {CFG.model_name}')
    print(f'backbone {CFG.backbone}')
    model = CustomModel(CFG, weight)

    return model.cuda()

class Data_loader(Dataset):
     
    def __init__(self,paths,is_label):
        self.paths=paths
        self.paths.sort()
        self.is_label=is_label
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,index):
        img = cv2.imread(self.paths[index],cv2.IMREAD_GRAYSCALE)
        img = helper.to_1024_1024(img, CFG, image_size = CFG.image_size )

        img = tc.from_numpy(img.copy())
        if self.is_label:
            img=(img!=0).to(tc.uint8)*255
        else:
            img=img.to(tc.uint8)
        return img

def load_data(paths, is_label=False):
    data_loader = Data_loader(paths, is_label)
    data_loader = DataLoader(data_loader, batch_size=16, num_workers=2)  
    data=[]
    for x in tqdm(data_loader):
        data.append(x)
    x = tc.cat(data,dim=0)
    del data
    if not is_label:
        TH = x.reshape(-1).numpy()
        index = -int(len(TH) * CFG.chopping_percentile)
        TH:int = np.partition(TH, index)[index]
        x[x>TH] = int(TH)
        TH = x.reshape(-1).numpy()
        index = -int(len(TH) * CFG.chopping_percentile)
        TH:int = np.partition(TH, -index)[-index]
        x[x<TH] = int(TH)
        x=(helper.min_max_normalization(x.to(tc.float16)[None])[0]*255).to(tc.uint8)
    return x

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.sigmoid()   
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class Kaggld_Dataset(Dataset):
    def __init__(self,x:list,y:list,arg=False):
        super(Dataset,self).__init__()
        self.x = x
        self.y = y
        self.image_size = CFG.image_size
        self.in_chans = CFG.in_chans
        self.arg = arg
        if arg:
            self.transform = CFG.train_aug
        else: 
            self.transform = CFG.valid_aug

    def __len__(self) -> int:
        return sum([y.shape[0] - self.in_chans for y in self.y])
    
    def __getitem__(self,index):
        i=0
        for x in self.x:
            if index > x.shape[0]-self.in_chans:
                index -= x.shape[0]-self.in_chans
                i += 1
            else:
                break
        x=self.x[i]
        y=self.y[i]
        
        print(f'x.shape[1] ={x.shape[1]}    x.shape[2]={x.shape[2]}')
        
        x_index= (x.shape[1] - self.image_size)//2
        y_index= (x.shape[2] - self.image_size)//2
        x=x[index:index+self.in_chans, x_index:x_index+self.image_size, y_index:y_index+self.image_size]
        y=y[index+self.in_chans//2, x_index:x_index+self.image_size, y_index:y_index+self.image_size]

        data = self.transform(image=x.numpy().transpose(1,2,0), mask=y.numpy())
        x = data['image']
        y = data['mask']>=127
        if self.arg:
            i=np.random.randint(4)
            x=x.rot90(i,dims=(1,2))
            y=y.rot90(i,dims=(0,1))
            for i in range(3):
                if np.random.randint(2):
                    x=x.flip(dims=(i,))
                    if i>=1:
                        y=y.flip(dims=(i-1,))
        return x, y

if __name__ == "__main__":
    train_x=[]
    train_y=[]

    root_path="D:/data/"
    parhs=["D:/data/train/kidney_1_dense"]
    for i,path in enumerate(parhs):
        if path=="D:/data/train/kidney_3_dense":
            continue
        x=load_data(glob(f"{path}/images/*"),is_label=False)
        y=load_data(glob(f"{path}/labels/*"),is_label=True)
        train_x.append(x)
        train_y.append(y)
        train_x.append(x.permute(1,2,0))
        train_y.append(y.permute(1,2,0))
        train_x.append(x.permute(2,0,1))
        train_y.append(y.permute(2,0,1))

    path1 = "D:/data/train/kidney_3_sparse"
    path2 = "D:/data/train/kidney_3_dense"
    paths_y = glob(f"{path2}/labels/*")
    paths_x = [x.replace("labels","images").replace("dense","sparse") for x in paths_y]
    val_x, val_y = load_data(paths_x,is_label=False), load_data(paths_y,is_label=True)

    tc.backends.cudnn.enabled = True
    tc.backends.cudnn.benchmark = True
        
    train_dataset = Kaggld_Dataset(train_x,train_y,arg=True)
    train_dataset = DataLoader(train_dataset, batch_size=CFG.train_batch_size ,num_workers=2, shuffle=True, pin_memory=True)
    val_dataset = Kaggld_Dataset([val_x],[val_y])
    val_dataset = DataLoader(val_dataset, batch_size=CFG.valid_batch_size, num_workers=2, shuffle=False, pin_memory=True)

    model = build_model()
    model = DataParallel(model)

    loss_fc = DiceLoss()
    optimizer = tc.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = tc.cuda.amp.GradScaler()
    scheduler = tc.optim.lr_scheduler.OneCycleLR(optimizer,
                                                 max_lr = CFG.lr,
                                                 steps_per_epoch = len(train_dataset), 
                                                 epochs=CFG.epochs+1,
                                                 pct_start = 0.1,)
    
    for epoch in range(CFG.epochs):
        model.train()
        time = tqdm(range(len(train_dataset)))
        loss, scores = 0, 0
        for i,(x,y) in enumerate(train_dataset):
            x = x.cuda().to(tc.float32)
            y = y.cuda().to(tc.float32)
            x = helper.norm_with_clip(x.reshape(-1,*x.shape[2:])).reshape(x.shape)
            x = helper.add_noise(x,max_randn_rate=0.5,x_already_normed=True)
            
            with autocast():
                pred = model(x)
                loss = loss_fc(pred,y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            # sub in different score function
            score = helper.dice_coef(pred.detach(),y)
            losss = (losss * i + loss.item())/(i + 1)
            scores = (scores * i + score)/(i + 1)
            time.set_description(f"epoch:{epoch},loss:{losss:.4f},score:{scores:.4f},lr{optimizer.param_groups[0]['lr']:.4e}")
            time.update()
            del loss, pred
        time.close()
        
        model.eval()
        time = tqdm(range(len(val_dataset)))
        val_losss = 0
        val_scores = 0
        resultssss = []
        for i,(x,y) in enumerate(val_dataset):
            x = x.cuda().to(tc.float32)
            y = y.cuda().to(tc.float32)
            x = helper.norm_with_clip(x.reshape(-1,*x.shape[2:])).reshape(x.shape)

            with autocast():
                with tc.no_grad():
                    pred = model(x)
                    loss = loss_fc(pred, y)

            score = helper.dice_coef(pred.detach(),y)
            val_losss = (val_losss * i + loss.item())/(i + 1)
            val_scores = (val_scores * i + score)/(i + 1)
            time.set_description(f"val-->loss:{val_losss:.4f},score:{val_scores:.4f}")
            time.update()

        time.close()
    tc.save(model.module.state_dict(),f"./{CFG.backbone}_{epoch}_loss{losss:.2f}_score{scores:.2f}_val_loss{val_losss:.2f}_val_score{val_scores:.2f}_midd_1024_final3.pt")

    time.close()



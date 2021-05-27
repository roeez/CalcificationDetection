import torch
from torch import nn
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import IPython
import os
from google.colab.patches import cv2_imshow
from google.colab import files
from IPython.display import clear_output 




SHEBA_MEAN = 0.1572722182674478
SHEBA_STD = 0.16270082671743363

class CBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBlock, self).__init__()
        assert out_channels%4==0
        self.conv3 = nn.Conv2d(in_channels, out_channels//4, 3, padding=3//2)
        self.conv5 = nn.Conv2d(in_channels, out_channels//4, 5, padding=5//2)
        self.conv7 = nn.Conv2d(in_channels, out_channels//4, 7, padding=7//2)
        self.conv9 = nn.Conv2d(in_channels, out_channels//4, 9, padding=9//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(self.bn(torch.cat([self.conv3(x), self.conv5(x), self.conv7(x), self.conv9(x)], 1)))

class NetC(nn.Module):

    def __init__(self, tag, kernel_size=9, skip_connections=True, batch_norm=True, kernel_depth_seed=4, network_depth=4, act_func=nn.ReLU(),
                 initializer=None):
        super(NetC, self).__init__()
        self.tag = tag
        self.block1 = CBlock(1, 4)
        self.block2 = CBlock(4, 16)
        self.block3 = CBlock(16, 32)
        self.block4 = CBlock(32, 64)
        self.block5 = CBlock(64, 128)
        self.pred = nn.Conv2d(128, 1, 5, padding=5//2)
    
    def forward(self, x):
        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x= self.block5(x)

        return self.pred(x)
    
model = NetC(tag='encoder')
model.eval()
model.load_state_dict(torch.load(os.path.join(__package__, 'C__00900.weights'), map_location=next(model.parameters()).device))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if 'cpu' in str(device):
    print("Computation will be very slow! to speed-up computation in the top menu: Runtime->Change runtime type->GPU")

model.to(device)

def left_mamm(mamm):
    if mamm[:, :200, ...].sum() < mamm[:, -200:, ...].sum():
            mamm[:, :, ...] = mamm[:, ::-1, ...]

    return mamm


def get_act_width(mamm):
    w = mamm.shape[1] // 3

    while mamm[:, w:].max() > 0:
        w += 1

    return w


def clean_mamm(mamm):
    background_val = 0
    mamm[:10, :, ...] = 0
    mamm[-10:, :, ...] = 0
    mamm[:, -10:, ...] = 0
        
    msk1 = (mamm[:, :, 0] == mamm[:, :, 1]) & (mamm[:, :, 1] == mamm[:, :, 2])
    mamm = mamm.mean(axis=2) * msk1
    msk = np.uint8((mamm > background_val) * 255)
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50)))

    comps = cv2.connectedComponentsWithStats(msk)

    common_label = np.argmax(comps[2][1:, cv2.CC_STAT_AREA]) + 1

    msk = (comps[1] == common_label).astype(np.uint8)

    mamm[:, :] = msk * mamm[:, :]

    return mamm


def cut_mamm(mamm, act_w):
    h = mamm.shape[0]
        # mamm[k] = v[:h - (h % 16), :act_w + (-act_w % 16)]
    mamm = mamm[:h, :act_w]

    # assert mamm['mamm'].shape[0] % 16 == mamm['mamm'].shape[1] % 16 == 0

    return mamm

def mamms_preprocess(mamm):
    mamm = left_mamm(mamm)
    mamm = clean_mamm(mamm)

    act_w = get_act_width(mamm)
    
    mamm = cut_mamm(mamm, act_w)

    return mamm

def predict(net, img):
    model_device = next(net.parameters()).device
    sig = nn.Sigmoid()
    toten = transforms.ToTensor()
    norm = transforms.Normalize(mean=[SHEBA_MEAN], std=[SHEBA_STD])
    if len(img.shape) == 2:
        img = img[..., None]
    img = norm(toten(img).float())[:1][None, ...].float()
    img = img.to(model_device)
    with torch.no_grad():
        pred = net(img)
    sig_pred = sig(pred)
    
    return sig_pred[0, 0].cpu().numpy()


def load_mamm(case_path, max_height=0, width=0, encoder=None):
    mamm = cv2.imread(case_path).astype(np.float32) / 255

    mamm = mamms_preprocess(mamm)

    return mamm

def upload_mamm():
    uploaded = files.upload()
    for k, v in uploaded.items():
        open(k, 'wb').write(v)
        break;

    mamm = cv2.imread(k).astype(np.float32) / 255

    mamm = mamms_preprocess(mamm)

    return mamm



def display_np_img(img):
    img = img.copy().astype('float')
    img -= img.min()
    img /= img.max()
    img = (255 * img).astype('uint8')

    cv2_imshow(img)


def update(img, x, y):
    tmp = mamm_w_heatmap
    tmp[...,2]= x*tmp[...,2] + y*result2
    display_np_img(tmp)

def show_mamm_w_boxes(processed_mamm, prediction, th=.5):
    result = (np.tile(processed_mamm[...,None], (1,1,3))*255).astype('uint8')
    bbs = np.zeros_like(result)
    cc = cv2.connectedComponentsWithStats((prediction>th).astype('uint8'), 8)
    for i in range(1, cc[0]):
        start_point = cc[2][i][0]-5, cc[2][i][1]-5 
        end_point = start_point[0] + cc[2][i][2]+10, start_point[1] + cc[2][i][3]+10
        cv2.rectangle(bbs, start_point, end_point ,(0,0,255), cv2.FILLED)
    clear_output()
    display_np_img(cv2.addWeighted(result, 1.0, bbs, .5, 1))



import argparse

from utils.utils import *
# from inference_utils.infer_utils import *
from models.model import Net3D
import torch
import os



parser = argparse.ArgumentParser(description='distinguish between malignant and osteoporotic fractures.')
parser.add_argument('-path',
                    help='path to nifti image')

parser.add_argument('-vert', action='store', dest='vertebrae',
                    type=str, nargs='*', default=['L1', 'L2'],
                    help="Examples: -v item1 item2, -i item3")
args = parser.parse_args()
print(args)
# print(args.p)
# print(args.vertebrae)

verts_dict = {'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7, 'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13, 'T7': 14,
 'T8': 15, 'T9': 16, 'T10': 17, 'T11': 18, 'T12': 19, 'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24, 'L6': 25}
inverse_dict = {v: k for k, v in verts_dict.items()}

img_data, ctd = read_datapoint(args.path)
verts_to_gen = [verts_dict[x] for x in args.vertebrae if x in verts_dict.keys()]
vertebrae_dict = get_3d_data(img_data, ctd, verts_to_gen)

if  torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_rng_state(torch.cuda.get_rng_state())
    torch.backends.cudnn.deterministic = True
else:
    device = torch.device('cpu')

model = Net3D(
    projector_base = 16,
    classifier_channels=(256, 128, 64),
    classes=2)

model.to(device)
model_path = os.path.join(os.path.dirname(__file__),'network_weights/model.tar')
state   = torch.load(model_path, map_location=device)
state_dict 	= state['state_dict']
model.load_state_dict(state_dict)

thresh_ost= 0.95
thresh_mal= 0.85

prediction_dict = {
        'FileID': os.path.basename(args.path)  
    }

for k, v in vertebrae_dict.items():
    if k<8:
        continue
    img = v['im']
    img = img.transpose((2, 0, 1))
    img = torch.tensor(img)
    img = torch.unsqueeze(torch.unsqueeze(img,0).type(torch.FloatTensor),0).to(device)
    
    
    model.eval()
    
    with torch.no_grad():
        logits = model.forward(img)
        softmaxed = torch.softmax(logits, dim=1)
        _,pred = torch.max(softmaxed, 1)

        pred = pred.detach().cpu().numpy()[0]
        
        proba =softmaxed[0].detach().cpu()
        uncert = 0
        if pred==1 and proba[1] < thresh_mal:
            uncert=1
        elif pred==0 and proba[0] < thresh_ost:
            uncert=1
            
        prediction_dict[inverse_dict[k]] = {'pred': pred, 'uncert': uncert}
        save_json(prediction_dict,args.path.replace('rawdata','derivatives').replace('_ct.nii.gz','_pred.json'))

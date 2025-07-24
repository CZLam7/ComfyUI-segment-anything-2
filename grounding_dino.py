import os
import torch
import comfy.model_management as mm
import folder_paths
from PIL import Image

# Local Grounding DINO utilities from ShilongLiu's repo
from local_groundingdino.util.slconfig import SLConfig
from local_groundingdino.util.utils import clean_state_dict
from local_groundingdino.models import build_model
from local_groundingdino.datasets import transforms as T
print("[DEBUG] grounding_dino.py loaded")


# Map of available DINO model names to their HF URLs
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url":  "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url":  "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth",
    },
}

# Helper: download or locate a file under models_dir

def get_local_filepath(url, dirname):
    parsed = os.path.basename(url)
    dest_dir = os.path.join(folder_paths.models_dir, dirname)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, parsed)
    if not os.path.exists(dest):
        from torch.hub import download_url_to_file
        download_url_to_file(url, dest)
    return dest

# Load a GroundingDINO model instance

def load_groundingdino_model(model_name):
    cfg_info = groundingdino_model_list[model_name]
    cfg_path = get_local_filepath(cfg_info["config_url"], "grounding-dino")
    model_path = get_local_filepath(cfg_info["model_url"],  "grounding-dino")

    # parse config and build model
    dino_args = SLConfig.fromfile(cfg_path)
    if dino_args.text_encoder_type == 'bert-base-uncased':
        # use local HF bert if available
        bert_dir = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
        dino_args.text_encoder_type = bert_dir if os.path.isdir(bert_dir) else 'bert-base-uncased'

    dino = build_model(dino_args)
    checkpoint = torch.load(model_path, map_location='cpu')
    dino.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    dino.to(device=mm.get_torch_device())
    dino.eval()
    return dino

# Run prediction: image (PIL), prompt (str), threshold (float)

def groundingdino_predict(dino, pil_image, prompt, box_threshold):
    # prepare image tensor
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    img_tensor, _ = transform(pil_image, None)
    img_tensor = img_tensor.to(mm.get_torch_device())

    # inference
    with torch.no_grad():
        outputs = dino(img_tensor[None], captions=[prompt.lower().strip() + '.'])
    logits = outputs['pred_logits'].sigmoid()[0]  # (nq,256)
    boxes  = outputs['pred_boxes'][0]             # (nq,4)
    mask = logits.max(dim=1)[0] > box_threshold
    boxes = boxes[mask]

    # convert center-based to corner coords
    H, W = pil_image.size[1], pil_image.size[0]
    boxes = boxes.cpu() * torch.Tensor([W,H,W,H])
    # xywh -> x1y1x2y2
    xy, wh = boxes[:,:2], boxes[:,2:]
    xy1 = xy - wh/2
    xy2 = xy1 + wh
    boxes = torch.cat([xy1, xy2], dim=1)
    return boxes.cpu().numpy().tolist()

# ComfyUI node: load + predict in one

class GroundingDINOPredict:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":      ("IMAGE",),   # HxWx3 float in [0,1]
                "prompt":     ("STRING", {"default":"object"}),
                "model_name": (list(groundingdino_model_list.keys()), {"default":"GroundingDINO_SwinT_OGC"}),
                "threshold":  ("FLOAT", {"default":0.3, "min":0.0, "max":1.0, "step":0.01}),
                "device":     (["cuda","cpu","mps"], {"default":"cuda"}),
            }
        }

    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("boxes",)
    FUNCTION = "predict"
    CATEGORY = "GroundingDINO"

    def predict(self, image, prompt, model_name, threshold, device):
        # convert ComfyUI IMAGE -> PIL
        arr = (image.numpy().squeeze() * 255).astype('uint8')
        pil = Image.fromarray(arr)

        # set device
        torch_device = torch.device(device if device != 'cuda' else 'cuda')

        # load and predict
        dino = load_groundingdino_model(model_name).to(torch_device)
        boxes = groundingdino_predict(dino, pil, prompt, threshold)
        return (boxes,)

NODE_CLASS_MAPPINGS = {
    "GroundingDinoPredict": GroundingDINOPredict
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroundingDinoPredict": "Grounding DINO Predictor"
}

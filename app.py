import gradio as gr
from torchvision import transforms
from torchvision.models.resnet import resnet18
import torch.nn as nn
import torch, warnings
warnings.filterwarnings("ignore")
from PIL import Image

labels = [
    'dandere',
    'deredere',
    'himdere',
    'kuudere',
    'tsundere',
    'yandere',
    'yangire'
]

resnet = resnet18(pretrained=False)
resnet.fc = nn.Linear(resnet.fc.in_features, len(labels))
resnet.load_state_dict(torch.load('./assets/derenet18.pth', map_location=torch.device('cpu')))
transform = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
resnet.eval()

def predict(img):
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    preds = nn.functional.softmax(resnet(img)[0], dim=0)
    return {labels[i]: float(preds[i]) for i in range(len(labels))}

title = "Dere Net"
description = "Demo for Dere Net. To use it, simply upload your waifu, or click one of the waifus below to load them."


inputs = gr.inputs.Image()
outputs = gr.outputs.Label(num_top_classes=4)
gr.Interface(
    fn=predict, 
    inputs=inputs,
    outputs=outputs,
    title=title, 
    description=description, 
    allow_flagging = False,
    layout = 'horizontal',
    examples = [['./assets/mikuex.jpg'],['./assets/zerotwo.jpg'],['./assets/mikasaex.jpg']]
    ).launch()

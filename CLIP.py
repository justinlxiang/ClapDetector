import numpy as np
import torch
from pkg_resources import packaging
import clip
import os
from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"  # Force the use of CPU regardless of CUDA availability

model, preprocess = clip.load("ViT-L/14@336px", device = device)
model.eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

images = []
texts = []

descriptions = ["This is a person not clapping with their hands apart", "This is a person clapping with their palms touching"]
text = clip.tokenize(descriptions).to(device)

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

positive_correct = []
negative_correct = []
correct = []

labels_folder = "labels_folder"
for participant in sorted(os.listdir(labels_folder)):
    for session in sorted(os.listdir(os.path.join(labels_folder, participant))):
        for textfile in os.listdir(os.path.join(labels_folder, participant, session)):
            with open(os.path.join(labels_folder, participant, session,textfile), 'r') as file:
                for line in sorted(file, key=lambda x: int(x.split('_')[-1].split('.')[0])):
                    label, image_path = line.strip().split()
                    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                    images.append(image)
                    
                    # image_features = model.encode_image(image)
                    # text_features = model.encode_text(text)

                    logits_per_image, logits_per_text = model(image, text)
                    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

                    pred = descriptions[argmax(list(probs)[0])]
                    print(image_path)
                    print("Prediction: ", pred)
                    print("Actual: ", descriptions[int(label)])
                    print()

                    if pred == descriptions[int(label)]:
                        if label == 1:
                            positive_correct.append(1)
                        elif label == 0:
                            negative_correct.append(1)
                        correct.append(1)
                    else:
                        correct.append(0)
                        positive_correct.append(0)
                        negative_correct.append(0)

print('accuracy on claps is :' + str(sum(positive_correct)/len(positive_correct)))
print('accuracy on non claps is :' + str(sum(negative_correct)/len(negative_correct)))
print('accuracy on all is : ' + str(sum(correct)/len(correct)))

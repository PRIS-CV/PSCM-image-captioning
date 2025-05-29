import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/coco/oscar_split_{clip_model_name}_train.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('./data/coco/annotations/train_caption.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        filename = f"./data/coco/train2014/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"./data/coco/val2014/val2014/COCO_val2014_{int(img_id):012d}.jpg"

        image = io.imread(filename) #用来读取图像

        # print("imread后")
        # print(image.shape)

        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)#对图像进行预处理

        # print("unsqueeze后：")
        # print(image.shape)

        with torch.no_grad(): #不进行反向传播
            prefix = clip_model.encode_image(image).cpu() #对图像进行特征提取

            # print("打印前缀的维度：")
            # print(prefix.shape)

        d["clip_embedding"] = i

        print(f'clip_embedding:\n{"clip_embedding"}\n')

        all_embeddings.append(prefix)
        all_captions.append(d)

        print(f'all_embeddings:\n{all_embeddings.len}\n')
        print(f'all_captions:\n{all_captions}\n')


        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f: #将图像存储到文件中
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
        

        #不进入循环，只是展示一张图片的信息
        break

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))

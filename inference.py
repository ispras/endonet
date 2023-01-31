import argparse
import cv2
import numpy as np
import torch

from dependencies import *


if __name__ == '__main__':

    if torch.cuda.is_available():
        DEVICE = 'cuda:0'
    else:
        DEVICE = 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='Path to image')
    parser.add_argument('--config_path', type=str, default='supplementary/pretrained_config.json', help='Path to config file')
    parser.add_argument('--model_path', type=str, default='supplementary/pretrained_checkpoint.pth', help='Path to model')
    args = parser.parse_args()

    dataset, config = make_dataset_from_image(args.image_path, args.config_path)
    model = define_model(config['model']).to(device=DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        image_input = dataset[0]['image'].unsqueeze(0).to(device=DEVICE)
        output = model(image_input)
        keypoints = output['keypoints']
        heatmaps = output['heatmaps'].cpu().numpy()

        heatmaps_stroma = heatmaps[0, 0, :, :]
        heatmaps_epithelium = heatmaps[0, 1, :, :]

        with open('output.txt', 'w') as f:
            f.write('X Y Class\n')
            for line in keypoints:
                f.write(' '.join(list(map(str, list(map(int, line))))[1:]) + '\n')

        heatmaps_stroma = np.clip(heatmaps_stroma, 0, 1)
        heatmaps_stroma = np.uint8(heatmaps_stroma * 255)
        cv2.imwrite('heatmaps_stroma.png', heatmaps_stroma)

        heatmaps_epithelium = np.clip(heatmaps_epithelium, 0, 1)
        heatmaps_epithelium = np.uint8(heatmaps_epithelium * 255)
        cv2.imwrite('heatmaps_epithelium.png', heatmaps_epithelium)


    remove_dummy_files()
import json
import numpy as np
import torch
import scipy.stats as sps

from dependencies import *


def calculate_confidence_interval(differences):
    """
    Calculates confidence interval for difference between metrics of baseline and pretrained models

    Args:
        differences (list): list of differences between metrics of baseline and pretrained models

    Returns:
        dict: For each of (general, stroma, epithelium) returns tuple of left and right bounds of confidence interval
    """    
    diffs, diffs_stroma, diffs_epith = differences
    cis_gen = []
    cis_str = []
    cis_epi = []
    for _ in range(10000):
        cis_gen.append(sps.bootstrap(diffs, np.mean, confidence_level=0.95, n_resamples=10000).confidence_interval)
        cis_str.append(sps.bootstrap(diffs_stroma, np.mean, confidence_level=0.95, n_resamples=10000).confidence_interval)
        cis_epi.append(sps.bootstrap(diffs_epith, np.mean, confidence_level=0.95, n_resamples=10000).confidence_interval)
    cis_gen, cis_str, cis_epi = np.array(cis_gen), np.array(cis_str), np.array(cis_epi)
    ci_gen_left, ci_gen_right = np.mean(cis_gen[:, 0]), np.mean(cis_gen[:, 1])
    ci_str_left, ci_str_right = np.mean(cis_str[:, 0]), np.mean(cis_str[:, 1])
    ci_epi_left, ci_epi_right = np.mean(cis_epi[:, 0]), np.mean(cis_epi[:, 1])

    return {'general': (ci_gen_left, ci_gen_right),
            'stroma': (ci_str_left, ci_str_right),
            'epithelium': (ci_epi_left, ci_epi_right)}


def calculate_metrics_by_pathes(config_path, checkpoint_path, test_ds_path, device='cpu', batch_size=1, num_workers=1):
    """
    Calculates metrics for model by pathes to config, checkpoint and test dataset

    Args:
        config_path (str): Path to config file
        checkpoint_path (str): Path to checkpoint file
        test_ds_path (str): Path to .yml that describes test dataset
        device (str, optional): Device used to calculate outputs of model. Defaults to 'cpu'.
        batch_size (int, optional): Number of images in a single batch. Defaults to 1.
        num_workers (int, optional): Number of workers used in dataloader. Defaults to 1.

    Returns:
        tuple: Tuple of arrays of APs for each class and general mAP
    """    
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = define_model(config['model']).to(device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    criterion = parse_eval_config(config['eval'])

    dataset = get_heatmaps_dataset(test_ds_path, config)
    dataloader = DataLoader(dataset,
                            collate_fn=dataset.collate,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers)

    metrics = []
    with torch.no_grad():
        for i in dataloader:
            criterion.reset()
            output = model(i['image'].to(device=device))
            criterion.update(output, i)
            criterion.compute()
            metrics.append(criterion.value)

    class_0 = np.array([x[0] for x in metrics])
    class_1 = np.array([x[1] for x in metrics])
    mAP = np.array([x[2] for x in metrics])

    return (class_0, class_1, mAP)


if __name__ == '__main__':
    if torch.cuda.is_available():
        DEVICE = 'cuda:0'
    else:
        DEVICE = 'cpu'

    PATH_TO_BASELINE_CONFIG = 'supplementary/baseline_cfg.json'
    PATH_TO_PRETRAINED_CONFIG = 'supplementary/pretrained_config.json'

    PATH_TO_BASELINE_CHECKPOINT = 'supplementary/baseline_checkpoint.pth' 
    PATH_TO_PRETRAINED_CHECKPOINT = 'supplementary/pretrained_checkpoint.pth'

    PATH_TO_TEST_DS = 'supplementary/test.yaml'
    BATCH_SIZE = 7
    NUM_WORKERS = 4

    baseline_class_0, baseline_class_1, baseline_map = calculate_metrics_by_pathes(PATH_TO_BASELINE_CONFIG,
                        PATH_TO_BASELINE_CHECKPOINT, PATH_TO_TEST_DS, DEVICE, BATCH_SIZE, NUM_WORKERS)

    pretrained_class_0, pretrained_class_1, pretrained_map = calculate_metrics_by_pathes(PATH_TO_PRETRAINED_CONFIG,
                        PATH_TO_PRETRAINED_CHECKPOINT, PATH_TO_TEST_DS, DEVICE, BATCH_SIZE, NUM_WORKERS)


    print(f'stroma AP, baseline_model: {baseline_class_0.mean():23}, pretrained_model: {pretrained_class_0.mean():23}')
    print(f'epithelium AP, baseline_model: {baseline_class_1.mean():19}, pretrained_model: {pretrained_class_1.mean():23}')
    print(f'general mAP, baseline_model: {baseline_map.mean():21}, pretrained_model: {pretrained_map.mean():23}')
    
    diffs = (pretrained_map-baseline_map, )
    diffs_stroma = (pretrained_class_0 - baseline_class_0, )
    diffs_epith = (pretrained_class_1 - baseline_class_1, )

    conf_intervals = calculate_confidence_interval([diffs, diffs_stroma, diffs_epith])

    ci_general = conf_intervals['general']
    ci_epith = conf_intervals['epithelium']
    ci_stroma = conf_intervals['stroma']
    
    print('Confidence intervals for differences in means:')
    print('general CI for diffs in means: {}'.format([ci_general[0], ci_general[1]]))
    print('stroma CI for diffs in means:  {}'.format([ci_stroma[0], ci_stroma[1]]))
    print('epith CI for diffs in means:   {}'.format([ci_epith[0], ci_epith[1]]))
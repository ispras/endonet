import json
import shutil
import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from matplotlib import patches, cm
from segmentation_models_pytorch import UnetPlusPlus
import matplotlib.pyplot as plt
import os
import warnings
from bidict import bidict
from torch import nn


class SimilarityMeasure:
    """
    Base class for similarity measures
    """

    def measure(self, object1, object2):
        """
        Returns the measure value between two objects
        """
        return 0

    def matrix(self, container1, container2):
        """
        Returns the matrix of measure values between two sets of objects
        Sometimes can be implemented in a faster way than making couplewise measurements
        """
        matrix = np.zeros((len(container1), len(container2)))
        for i, object1 in enumerate(container1):
            for j, object2 in enumerate(container2):
                matrix[i, j] = self.measure(object1, object2)
        return matrix


class Minkovsky2DSimilarity(SimilarityMeasure):
    def __init__(self, p=2, scale=1.0):
        self.p = p
        self.scale = scale

    def measure(self, point1, point2):
        x_diff = point1.x_coords()[0] - point2.x_coords()[0]
        y_diff = point1.y_coords()[0] - point2.y_coords()[0]
        diffs = np.hstack([np.abs(x_diff), np.abs(y_diff)])
        powers = np.power(diffs, self.p)
        distance = np.power(powers.sum(), 1 / self.p) / self.scale

        return distance

    def matrix(self, points1, points2):
        coords1 = np.vstack([points1.x_coords(), points1.y_coords()])
        coords2 = np.vstack([points2.x_coords(), points2.y_coords()])

        diffs = np.abs(coords1[:, :, np.newaxis] - coords2[:, np.newaxis, :])

        powers = np.power(diffs, self.p)
        matrix = np.power(powers.sum(axis=0), 1 / self.p) / self.scale

        return matrix


class KPSimilarity(Minkovsky2DSimilarity):
    """
    Keypoints similarity
    """

    def __init__(self, p=2, scale=1.0, class_agnostic=True):
        super().__init__(p, scale)
        self.class_agnostic = class_agnostic

    def _exp_square(self, arr):
        return np.exp(-np.power(arr, 2) / 2.0)

    def measure(self, point1, point2):
        distance = super().measure(point1, point2)
        return self._exp_square(distance)

    def matrix(self, points1, points2):
        distance = super().matrix(points1, points2)
        matrix = self._exp_square(distance)
        if not self.class_agnostic:
            class_matrix = points1.classes().reshape(-1, 1) == points2.classes().reshape(1, -1)
            matrix = matrix * class_matrix
        return matrix


def targets_mapping(
        similarity, sim_thresh, targets_gt, targets_pred, scores, multiclass=True
):
    """
    Maps ground truth targets to predictions for a single class.

    Parameters
    ----------
    similarity: endoanalysis.similarity.SimilarityMeasure
        a measure of similarity between targets
    sim_thresh: list float
        threshold for similarity between ground truth and prediction targets.
        If two targets have similarity below this threshold, they will not be mapped even if there are no better candidates
    targets_gt: endoanalysis.targets.ImageTargets
        ground truth targets (like keypoints or bboxes) for an image. Must be compatible with similarity.
    targets_pred: endoanalysis.targets.ImageTargets
        prediction targets (like keypoints or bboxes) for an image. Must be compatible with similarity.
    scores: npdarray
        scores (confidences) of the predictions. Must be the same length as targets_pred.
        The predictions with higher score have the priority in mapping
    multiclass : bool
        If True, only the targets with coinsiding class labels will be matched.


    Returns
    -------
    gt_to_pred: bidict.bidict
        mapping from ground truth targets to predictions
    gt_is_tp: ndarray of bool
        the array with true positive labels from ground truth targets
    pred_is_tp: ndarray of bool
        the array with true positive labels from ground pred     targets

    See also
    --------
    https://www.kaggle.com/chenyc15/mean-average-precision-metric
    """

    if len(targets_pred) != len(scores):
        raise ValueError("targets_pred and scores should have the same length")

    preds_ids_sorted = np.argsort(scores)[::-1]
    sim_matrix = similarity.matrix(targets_gt, targets_pred)
    gt_is_tp = np.zeros((len(targets_gt)), dtype=bool)
    pred_is_tp = np.zeros((len(targets_pred)), dtype=bool)
    gt_to_pred = bidict()

    if multiclass:
        classes_pred = targets_pred.classes()
        classes_gt = targets_gt.classes()
    else:
        classes_pred = np.zeros(len(targets_pred), dtype=int)
        classes_gt = np.zeros(len(targets_gt), dtype=int)

    for pred_j in preds_ids_sorted:
        pred_matched = False

        sims_for_pred = sim_matrix[:, pred_j]
        gt_ids_sorted = np.argsort(sims_for_pred)

        for gt_i in gt_ids_sorted:
            if (
                    sim_matrix[gt_i, pred_j] > sim_thresh
                    and not gt_is_tp[gt_i]
                    and not pred_matched
                    and classes_pred[pred_j] == classes_gt[gt_i]
            ):
                pred_matched = True
                pred_is_tp[pred_j] = True
                gt_is_tp[gt_i] = True
                gt_to_pred[gt_i] = pred_j

    return gt_to_pred, gt_is_tp, pred_is_tp


def interpolate_PR_curve(precisions, recalls):
    """
    Eliminates "wiggles" on the PR curve.

    Parameters
    ----------
    precisions: ndarray of float
        precision values
    recalls: ndarray of float
        recall values

    Returns
    -------
    precisions_interpolated:
        interpolated precision values with correspondence to raw precisions
    break_points:
        breaking points of the interpolated plot
    """
    last_precision = precisions[-1]
    precisions_interpolated = np.zeros_like(precisions)

    precisions_range = len(precisions)
    ids_to_change = []
    break_points = []

    for idx, precision in enumerate(precisions[::-1]):
        true_idx = precisions_range - idx - 1

        if precision < last_precision:
            ids_to_change.append(true_idx)
        else:
            precisions_interpolated[ids_to_change] = last_precision
            break_points.append(true_idx)

            ids_to_change = [true_idx]
            last_precision = precision

        if (
                len(break_points) > 1
                and recalls[break_points[-1]] == recalls[break_points[-2]]
        ):
            break_points.pop(-2)
        if (
                len(break_points) > 1
                and precisions[break_points[-1]] == precisions[break_points[-2]]
        ):
            break_points.pop(-1)

    if break_points[-1] == 0:
        break_points.pop()

    precisions_interpolated[ids_to_change] = last_precision
    return precisions_interpolated, break_points


def pr_curve(pred_is_tp, confidences, total_positives):
    """
    Calculates precision and recall values for PR-curve.

    Parameters
    ----------
    pred_is_tp: ndarray of bool
        true positive flags for predictions. If prediction is not true positive, it is considered to be false positive.
    confidences: ndarray of float
        predictions confidences
    total_positives: int
        total number of true positives

    Returns
    -------
    precisions: ndarray of float
        precision values
    recalls: ndarray of float
        recall values

    Note
    ----
    The PR-curve is defeined as follows.

    First all targets are sorted by their confidences in the descending order.
    Than the precision and recall values are calculated only for the nost confident target,
    than for two most confident targtes and so on.

    """
    pred_is_tp = np.copy(pred_is_tp)
    confidences = np.copy(confidences)

    ids_sorted = np.argsort(confidences)
    confidences = confidences[ids_sorted]
    pred_is_tp = pred_is_tp[ids_sorted]

    if not total_positives:
        raise Exception("No grounds truth labels present")

    total_preds = len(confidences)
    precisions = np.zeros((total_preds))
    recalls = np.zeros((total_preds))
    current_tps = 0

    for idx, is_tp in enumerate(pred_is_tp):
        if is_tp:
            current_tps += 1
        recalls[idx] = current_tps / total_positives
        precisions[idx] = current_tps / (idx + 1)

    return precisions, recalls


def avg_precisions_from_composed(
        confidences, pred_is_tp, gt_classes, pred_classes, class_labels
):
    """
    Calculates average precisions (AP) for different classes from composed confidences and labels.

    Parameters
    ----------
    confidences: ndarray of float
        concatenated confidences of the targets
    pred_is_tp: ndarray of bool
        concatenated true positive flags for predictions. If prediction is not true positive, it is considered to be false positive.
    gt_classes: ndarray of float
        classes for ground truth targets
    pred_classes: ndarray of float
        classes for predicted targets
    class_labels: iterable of float
        class labels to consider
    """
    avg_precs = {}
    curves = {}

    for class_i in class_labels:
        ids = np.where(pred_classes == class_i)[0]
        total_positives_gt = np.sum(gt_classes == class_i)
        total_positive_pred = np.sum(pred_is_tp[ids])
        if total_positives_gt == 0:
            warnings.warn("No true positives for class %i" % class_i)
            avg_precs[class_i] = None
        elif total_positive_pred == 0:
            avg_precs[class_i] = 0.0
            curves[class_i] = {
                "recalls": [0.0],
                "precisions": [0.0],
                "interpolated": [0.0],
            }
        else:
            precisions, recalls = pr_curve(
                pred_is_tp[ids], confidences[ids], total_positives_gt
            )
            precisions_interpolated, break_points = interpolate_PR_curve(
                precisions, recalls
            )
            avg_precs[class_i] = np.sum(
                precisions_interpolated[break_points]
                * (recalls[break_points] - recalls[break_points[1:] + [0]])
            )

            curves[class_i] = {
                "recalls": recalls,
                "precisions": precisions,
                "interpolated": precisions_interpolated,
            }

    return avg_precs, curves


def compose_tp_labels(
        similarity,
        sim_thresh,
        images_ids,
        targets_batched_pred,
        targets_batched_gt,
        confidences,
):
    """
    Maps predictions to groud truth targets for a batch of images.


    Parameters
    ----------
    similarity: endoanalysis.similarity.SimilarityMeasure
        a measure of similarity between targets
    sim_thresh: list float
        threshold for similarity between ground truth and prediction targets.
        If two targets have similarity below this threshold, they will not be mapped even if there are no better candidates
    images_ids: list of int
        ids of images to consider
    targets_batched_pred: endoanalysis.targets.ImageTargetsBatch
        prediction targets (like keypoints or bboxes) for a batch of images. Must be compatible with similarity.
    targets_batched_gt: endoanalysis.targets.ImageTargetsBatch
        ground truth targets (like keypoints or bboxes) for a batch of images. Must be compatible with similarity.
    confidences: ndarray of float
        confdences of the predcitions.

    Returns
    -------
    confidences: ndarray of float
        concatenated confidences of the targets
    pred_is_tp: ndarray of bool
        concatenated true positive flags for predictions. If prediction is not true positive, it is considered to be false positive.


    Note
    ----
    Only the images with ids form images_ids will be present in the result, with the same order as they
    appear in images_ids.
    """

    pred_is_tp = []
    confidences_to_stack = []

    if images_ids == "all":
        images_ids_get = targets_batched_gt.image_labels()
        images_ids_pred = targets_batched_pred.image_labels()
        images_ids = np.unique(
            np.concatenate([images_ids_get, images_ids_pred])
        ).astype(int)

    for image_i in images_ids:
        targets_gt = targets_batched_gt.from_image(image_i)
        targets_pred = targets_batched_pred.from_image(image_i)
        # classes_image = targets_batched_pred.from_image(image_i).classes()
        confidences_image = confidences[targets_batched_pred.image_labels() == image_i]
        confidences_to_stack.append(confidences_image)
        _, _, pred_is_tp_image = targets_mapping(
            similarity,
            sim_thresh,
            targets_gt,
            targets_pred,
            confidences_image,
            multiclass=True,
        )

        pred_is_tp.append(pred_is_tp_image)
        # classes.append(classes_image)

    if pred_is_tp:
        pred_is_tp = np.hstack(pred_is_tp)
        confidences = np.hstack(confidences_to_stack)
    else:
        pred_is_tp = np.empty(0)
        confidences = np.empty(0)
    # classes = np.hstack(classes)

    return confidences, pred_is_tp


class CumulativeMetric:
    def __init__(self, name="generic", name_group="generics", precision=6):
        self.name = name
        if name_group is None:
            name_group = name
        self.name_group = name_group
        self.precision = precision

    def reset():
        raise NotImplementedError

    def update():
        raise NotImplementedError

    def compute():
        raise NotImplementedError

    def get_logging_info(self):
        return self.name, self.name_group, self.value, self.precision

    def best_is_max(self):
        return True

    def min_max_factor(self):
        if self.best_is_max():
            return 1
        else:
            return -1

    def get_value(self):
        return self.value


class mAPmetric(CumulativeMetric):
    def __init__(
            self,
            similarity,
            class_labels,
            sim_thresh=0.5,
            name="mAP",
            name_group="mean_AP_meters",
            return_nans=False,
    ):
        super().__init__(name, name_group)
        self.class_labels = class_labels
        if type(class_labels) is int:
            self.class_labels = [class_labels]
        self.similarity = similarity
        self.sim_thresh = sim_thresh
        self.return_nans = return_nans
        self.reset()

    def reset(self):
        self.confidences = {}
        self.pred_is_tp = {}
        self.num_gt = {}
        self.classes_pred = {}
        self.value = 0

        for class_label in self.class_labels:
            self.confidences[class_label] = np.empty(0)
            self.pred_is_tp[class_label] = np.empty(0)
            self.num_gt[class_label] = 0
            self.classes_pred[class_label] = np.empty(0)

    def update_one_class(self, batch_pred, batch_gt, class_label):

        keypoints_pred = batch_pred["keypoints"]
        confidences = batch_pred["confidences"]
        keypoints_gt = batch_gt["keypoints"]

        keypoints_gt = keypoints_gt[keypoints_gt.classes() == class_label]
        confidences = confidences[keypoints_pred.classes() == class_label]
        keypoints_pred = keypoints_pred[keypoints_pred.classes() == class_label]

        confidences, pred_is_tp = compose_tp_labels(
            self.similarity,
            self.sim_thresh,
            "all",
            keypoints_pred,
            keypoints_gt,
            confidences,
        )

        self.pred_is_tp[class_label] = np.hstack(
            [self.pred_is_tp[class_label], pred_is_tp]
        )
        self.confidences[class_label] = np.hstack(
            [self.confidences[class_label], confidences]
        )
        self.num_gt[class_label] += len(keypoints_gt)

    def update(self, batch_pred, batch_gt):
        for class_label in self.class_labels:
            self.update_one_class(batch_pred, batch_gt, class_label)

    def compute_avg_precs_one_class(self, class_label):

        classes_pred = np.ones_like(self.pred_is_tp[class_label]) * class_label
        classes_gt = np.ones(self.num_gt[class_label]) * class_label

        avg_precisions, _ = avg_precisions_from_composed(
            self.confidences[class_label],
            self.pred_is_tp[class_label],
            classes_gt,
            classes_pred,
            [class_label],
        )

        return avg_precisions[class_label]

    def compute(self):
        mAP = 0
        APs = []
        num_not_none_classes = len(self.class_labels)
        for class_label in self.class_labels:
            AP = self.compute_avg_precs_one_class(class_label)
            if not AP is None:
                APs.append(AP)
                mAP += self.compute_avg_precs_one_class(class_label)
            else:
                num_not_none_classes -= 1

        if num_not_none_classes != 0:
            mAP /= num_not_none_classes
            self.value = float(mAP)
            APs.append(mAP)
            self.value = APs
        else:
            if self.return_nans:
                self.value = None
            else:
                self.value = 0
            warnings.warn("No objects to match! for %s" % str(self))

    def best_is_max(self):
        return True

    def __repr__(self):

        return " ".join(
            ["APmeter for the classses:"] + [str(x) for x in self.classes_pred]
        )


def check_images_and_labels_pathes(images_paths, labels_paths):
    if len(images_paths) != len(labels_paths):
        raise Exception("Numbers of images and labels are not equal")

    for image_path, labels_path in zip(images_paths, labels_paths):
        dirname_image = os.path.dirname(image_path)
        dirname_labels = os.path.dirname(labels_path)
        filename_image = os.path.basename(image_path)
        filename_labels = os.path.basename(labels_path)

        if ".".join(filename_image.split(".")[:-1]) != ".".join(
                filename_labels.split(".")[:-1]
        ):
            raise Exception(
                "Different dirnames found: \n %s\n  %s" % (images_paths, labels_paths)
            )


def extract_images_and_labels_paths(images_list_file, labels_list_file):
    images_list_dir = os.path.dirname(images_list_file)
    labels_list_dir = os.path.dirname(labels_list_file)

    with open(images_list_file, "r") as images_file:
        images = images_file.readlines()
        images = [
            os.path.normpath(os.path.join(images_list_dir, x.strip())) for x in images
        ]
    with open(labels_list_file, "r") as labels_file:
        labels = labels_file.readlines()
        labels = [
            os.path.normpath(os.path.join(labels_list_dir, x.strip())) for x in labels
        ]

    check_images_and_labels_pathes(images, labels)

    return images, labels


class KeypointsExtractor(nn.Module):
    def __init__(
            self, min_peak_value, pooling_scale, out_image_shape, supression_range, **kwargs
    ):
        super(KeypointsExtractor, self).__init__()
        self.out_image_shape = out_image_shape
        self.sim_thresh = 0.5
        self._peak_lower_bond = 0.01
        self._peak_greater_bond = 0.99
        self.set_params(min_peak_value, pooling_scale, supression_range)

    def set_params(self, min_peak_value, pooling_scale, supression_range):
        min_peak_value = np.max([min_peak_value, self._peak_lower_bond])
        self.min_peak_value = np.min([min_peak_value, self._peak_greater_bond])
        self.confidence_factor = 1.0 / (1.0 - self.min_peak_value)
        self.similarity = KPSimilarity(scale=supression_range)

        padding = int(np.floor(pooling_scale / 2).astype(int))
        self.pooler = nn.MaxPool2d(
            kernel_size=pooling_scale, padding=padding, stride=1)

    def keypoints_from_peaks_crossclass(self, heatmaps):
        num_classes = heatmaps.shape[1]
        heatmaps_cumulative, max_ids = heatmaps.max(axis=1, keepdim=False)
        pooled = self.pooler(heatmaps_cumulative)
        rel_maxs = pooled == heatmaps_cumulative
        image_ids, y_coords, x_coords = torch.where(rel_maxs)
        class_ids = max_ids[image_ids, y_coords, x_coords]
        peak_values = heatmaps[image_ids, class_ids, y_coords, x_coords]
        keypoints = (
            torch.vstack([image_ids, x_coords, y_coords, class_ids])
                .cpu()
                .numpy()
                .T.astype(float)
        )

        keypoints = KeypointsBatch(keypoints)

        return keypoints, peak_values.detach().cpu().numpy()

    def keypoints_from_peaks(self, heatmaps):
        """
        Extract keypoints from the heatmaps using max pooling.

        Parameters
        ----------
        heatmaps: ndarray
            images heatmaps, the size is (n_imgs, num_classes, y_size, x_size)

        Returns
        -------
        keypoints: endoanalysis.targets.KeypointsBatch
            extracted keypoints
        scores: torch.tensor
            the scores of the extracted keypoints
        """

        pooled = self.pooler(heatmaps)
        rel_maxs = pooled == heatmaps
        image_ids, class_ids, y_coords, x_coords = torch.where(rel_maxs)
        peak_values = heatmaps[image_ids, class_ids, y_coords, x_coords]
        keypoints = (
            torch.vstack([image_ids, x_coords, y_coords, class_ids])
                .cpu()
                .numpy()
                .T.astype(float)
        )

        keypoints = KeypointsBatch(keypoints)

        return keypoints, peak_values.detach().cpu().numpy()

    def supress(self, keypoints_batch, confidences_batch, batch_size):

        if not len(keypoints_batch):
            return keypoints_batch, confidences_batch

        keypoints_batch_supressed = []
        confidences_batch_supressed = []
        for image_i in range(batch_size):
            keypoints = keypoints_batch.from_image(image_i)

            confidences = confidences_batch[keypoints_batch.image_labels(
            ) == image_i]
            confidences_ids = np.argsort(confidences)
            kp_sorted = keypoints[confidences_ids][::-1]

            sim_matrix = self.similarity.matrix(kp_sorted, kp_sorted)
            overlap_matrix = sim_matrix > self.sim_thresh
            ids_to_keep = np.sum(np.triu(overlap_matrix, 1), axis=0) == 0
            keypoints_batch_supressed.append(kp_sorted[ids_to_keep])
            confidences_batch_supressed.append(
                confidences[confidences_ids][ids_to_keep]
            )

        keypoints_batch_supressed = keypoints_list_to_batch(
            keypoints_batch_supressed)
        confidences_batch_supressed = np.concatenate(
            confidences_batch_supressed)

        return keypoints_batch_supressed, confidences_batch_supressed

    def normalize_confidences(self, confidences):
        confidences -= self.min_peak_value
        confidences *= self.confidence_factor
        confidences[confidences > 1.0] = 1.0
        confidences[confidences < 0.0] = 0.0
        return confidences

    def forward(self, heatmaps):
        batch_size = heatmaps.shape[0]
        keypoints, peak_values = self.keypoints_from_peaks_crossclass(heatmaps)

        valid_peak_ids = peak_values > self.min_peak_value
        keypoints = keypoints[valid_peak_ids]
        confidences = peak_values[valid_peak_ids]
        confidences = self.normalize_confidences(confidences)
        keypoints, confidences = self.supress(
            keypoints, confidences, batch_size)

        keypoints = rescale_keypoints(
            keypoints,
            in_image_shape=heatmaps.shape[2:4],
            out_image_shape=self.out_image_shape,
        )

        return keypoints, confidences


class BaseDetector:
    """
    Base class for nuclei detectors.
    """

    def __init__(self):
        pass

    def detect_single(self, image):
        """
        Detect nuclei for single image

        Paramteres
        ----------
        image: ndarray
            input image, the shape is (C, H, W)

        Returns
        -------
        keypoints: ndarray
            detected keypoints
        confidences: ndarray
            confidences of the predictions
        """

        raise NotImplementedError()

    def detect_multi(self, images):
        """
        Detect nuclei for multiple images.

        Paramteres
        ----------
        images: iterable of ndarray
            input images, the shape of each image is (C, H, W)

        Returns
        -------
        images_keypoints: endoanalysis.targets.Keypoints
            detected keypoints for all images.
        images_confidences: ndarray
            concatenated confidences of the predictions

        """
        images_keypoints = []
        images_confidences = []
        for im in images:
            keypoints, confidences = self.detect_single(im)
            images_keypoints.append(keypoints)
            images_confidences.append(confidences)

        return keypoints_list_to_batch(images_keypoints), np.concatenate(
            images_confidences
        )


class HeatmapDetector(nn.Module, BaseDetector):
    """
    Nuclei detector based on heatmap model.
    A model converts images into several heatmaps (one heatmap per class),
    from which the keypoints are eextracted

    Parameters
    ----------
    preprocessor: nucleidet.data.preprocess.Preprocessor
        preprocessor instance wich converts images to the form requird byt the model
    heatmap_model: nucleidet.models.heatmap.HeatmapModel
        torch model to extract heatmaps from the images
    class_separator: nucleidet.detectors.heatmap.ClassSeparator
        class separator to force the keypoints from differen classes be present at
        the same spots.
    """

    def __init__(self,
                 backbone="Unet",
                 min_peak_value=0.,
                 pooling_scale=7.62,
                 out_image_shape=(512, 512),
                 supression_range=7,
                 **kwargs):
        super(HeatmapDetector, self).__init__()
        self.heatmap_model = globals()[backbone](**kwargs)
        self.keypoints_extractor = KeypointsExtractor(
            min_peak_value, pooling_scale, out_image_shape, supression_range)
        self.set_predict_keypoints(True)

    def set_predict_keypoints(self, flag):
        if flag:
            self.predict_keypoints = True
        else:
            self.predict_keypoints = False

    def topk(self, heatmaps, k=1):
        """
        Extract keypoints as topk values.

        Parameters
        ----------
        heatmaps: ndarray
            images heatmaps, the size is (n_imgs, num_classes, y_size, x_size)

        Returns
        -------
        keypoints: endoanalysis.targets.KeypointsBatch
            extracted keypoints
        scores: torch.tensor
            the scores of the extracted keypoints

        Note
        ----
        After this function a supression algorithm is usually used.
        """

        num_images, _, y_size, x_size = heatmaps.shape
        topk = torch.topk(heatmaps.flatten(
            start_dim=1, end_dim=3), k=k, dim=-1)
        topk_inds = topk.indices
        scores = topk.values

        classes = torch.floor(topk_inds / (y_size * x_size))
        topk_inds = topk_inds - classes * y_size * x_size
        y_coords = torch.floor((topk_inds) / x_size)
        topk_inds = topk_inds - y_coords * x_size
        x_coords = torch.floor((topk_inds))

        keypoints_pred = np.stack(
            [x_coords.numpy(), y_coords.numpy(), classes.numpy()], -1
        )
        images_ids = np.arange(num_images).repeat(k).reshape(-1, 1)
        keypoints_pred = np.hstack(
            [images_ids, keypoints_pred.reshape(num_images * k, -1)]
        )
        keypoints_pred = KeypointsBatch(keypoints_pred)

        return keypoints_pred, scores.reshape(-1)

    def detect_multi(self, images):
        _, keypoints, confidences = self(images)
        return keypoints, confidences

    def forward(self, images):
        heatmaps = self.heatmap_model(images)

        # heatmaps = self.class_separator(heatmaps)

        if self.predict_keypoints:
            keypoints, confidences = self.keypoints_extractor(heatmaps)
        else:
            keypoints = None
            confidences = None
        result = {"heatmaps": heatmaps,
                  "keypoints": keypoints, "confidences": confidences}
        return result


def agregate_images_and_labels_paths(images_lists, labels_lists):
    if type(images_lists) != type(labels_lists):
        raise Exception(
            "images_list_files and labels_list_file should have the same type"
        )

    if type(images_lists) != list:
        images_lists = [images_lists]
        labels_lists = [labels_lists]

    images_paths = []
    labels_paths = []
    for images_list_path, labels_list_path in zip(images_lists, labels_lists):
        images_paths_current, labels_paths_current = extract_images_and_labels_paths(
            images_list_path, labels_list_path
        )
        images_paths += images_paths_current
        labels_paths += labels_paths_current

    return images_paths, labels_paths


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_keypoints(file_path):
    """
    Load keypoints from a specific file as tuples

    Parameters
    ----------
    file_path : str
        path to the file with keypoints

    Returns
    -------
    keypoints : list of tuples
        list of keypoint tuples in format (x, y, obj_class)

    Note
    ----
    This function serves as helper for the pointdet.utils.dataset.PointsDataset class
    and probably should be moved there
    """

    keypoints = []

    with open(file_path, "r") as labels_file:
        for line in labels_file:
            line_contents = line.strip().split(" ")
            line_floated = tuple(int(float(x)) for x in line_contents)
            x_center, y_center, obj_class = tuple(line_floated)
            keypoint = x_center, y_center, obj_class
            keypoints.append(keypoint)

    return keypoints


def visualize_keypoints(
        image,
        keypoints,
        class_colors={x: cm.Set1(x) for x in range(10)},
        labels=None,
        fig=None,
        ax=None,
        circles_kwargs={"radius": 3, "alpha": 1.0, "ec": (0, 0, 0)}
):
    """
    Visualise keypoints on the image

    Parameters
    ----------
    image: ndarray
        input image. The shape must be (H, W, C)
    keypoints: endoanalysis.targets.keypoints
        keypoints to visualize
    class_colors:
        dictionary of colours for different classes
    labels: iterble of str
        texts to label the keypoints
    ax: matplotlib.axes._subplots.AxesSubplot
        plt subplot to draw image and keypoints.
        If not provided, will be generated and returned
    circles_kwargs: dict
        kwargs for circles pathces indicating the keypints

    Returns
    -------
    fig: matplotlib.figure.Figure or None
        plt figure object. if ax parametero is not provided, None will be returned
    ax: matplotlib.axes._subplots.AxesSubplot
        plt axis object with image and keypoints

    Example
    -------
    >>> visualize_keypoints(
    ...     image,
    ...     keypoints_image,
    ...     class_colors= {x: cm.Set1(x) for x in range(10)},
    ...     fig=fig,
    ...     ax=ax,
    ...     circles_kwargs={"radius": 2.5, "alpha": 1.0,  "linewidth": 2, 'ec': (0,0,0)}
    ...     )
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = None

    ax.imshow(image)
    ax.autoscale(False)

    x_coords = keypoints.x_coords()
    y_coords = keypoints.y_coords()
    classes = keypoints.classes()
    for i, (center_x, center_y, obj_class) in enumerate(zip(x_coords, y_coords, classes)):

        patch = patches.Circle((center_x, center_y), color=class_colors[obj_class], **circles_kwargs)
        ax.add_patch(patch)

        if labels is not None:
            ax.text(
                center_x,
                center_y,
                labels[i],
                c="b",
                fontweight="semibold",
            )

    return fig, ax


class PointsDataset:
    def __init__(
            self,
            images_list,
            labels_list,
            keypoints_dtype=np.float,
            class_colors={x: cm.Set1(x) for x in range(10)},
    ):

        self.keypoints_dtype = keypoints_dtype

        self.images_paths, self.labels_paths = agregate_images_and_labels_paths(
            images_list,
            labels_list,
        )
        self.class_colors = class_colors

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, x):

        image = load_image(self.images_paths[x])
        keypoints = load_keypoints(self.labels_paths[x])

        class_labels = [x[-1] for x in keypoints]
        keypoints_no_class = [x[:-1] for x in keypoints]

        keypoints = [
            np.array(y + (x,)) for x, y in zip(class_labels, keypoints_no_class)
        ]

        if keypoints:
            keypoints = np.stack(keypoints)
        else:
            keypoints = np.empty((0, 3))

        to_return = {"keypoints": Keypoints(keypoints.astype(self.keypoints_dtype))}

        to_return["image"] = image

        return to_return

    def visualize(
            self,
            x,
            show_labels=True,
            labels_kwargs={"radius": 3, "alpha": 1.0, "ec": (0, 0, 0)},
    ):

        sample = self[x]

        image = sample["image"]

        if show_labels:
            keypoints = sample["keypoints"]
        else:
            keypoints = Keypoints(np.empty((0, 3)))

        _ = visualize_keypoints(
            image,
            keypoints,
            class_colors=self.class_colors,
            circles_kwargs=labels_kwargs,
        )

    def collate(self, samples):

        images = [x["image"] for x in samples]
        keypoints_groups = [x["keypoints"] for x in samples]

        return_dict = {
            "image": np.stack(images, 0),
            "keypoints": keypoints_list_to_batch(keypoints_groups),
        }

        return return_dict


class ImageTargets(np.ndarray):
    """
    Subclass of ndarray designed to store the targets (objects on an image).
    This is the base class for  targets arrays of different types.

    Parameters
    ----------
    input_array : ndarray
        the array to create targets from.
        Must have float dtype and the shape (num_targets, param_num).
        If there is targets on the image, it should have the shape (0,).


    Methods
    -------
    conf_striped()
        returns the targets without the confidences.
    confidences()
        returns the targets confidences
    classes()
        returns the targets class labels
    specs()
        returns the strig describing targets format.

    Note
    ----
    The intialisations procedures (like checks of array shape and type) are done
    not in the  __array_filnalise__ as recommended in

    https://numpy.org/doc/stable/user/basics.subclassing.html

    but in the __init__ method. This is done to avoid unneccessary checks,
    for exmaple when slicing the  ImageTargets.
    """

    def __new__(cls, input_array):

        obj = np.asarray(input_array).view(cls)
        return obj

    def __init__(self, *args, **kwargs):
        self.param_num = 1
        self._check_shape()
        self._check_dtype()

    def _check_shape(self):

        if len(self.shape) != 2:
            raise Exception(
                " %s should have the shape (num_images, %i)"
                % (type(self), self.param_num)
            )
        elif self.shape[1] != self.param_num and self.shape[1] != 0:

            raise Exception(
                "The shape is %s, but the expected number of paramteres is %i"
                % (self.shape, self.param_num)
            )

    def _check_dtype(self):
        if self.dtype != float:
            raise Exception("ImageTargets entries should have dtype float")

    def conf_striped(self):
        """
        Returns the targets without the confidences.

        Returns
        -------
        targets_no_conf : ndarray of float
            2D array with confidences of shape (num_targets, param_num).
        """
        raise NotImplementedError()

    def confidences(self):
        """
        Returns the targets confidences.

        Returns
        -------
        confidences : ndarray of float
            1D array with confidences.
        """
        raise NotImplementedError()

    def classes(self):
        """
        Returns the class labels of the targets.

        Returns
        -------
        confidences : ndarray of int
            1D array with class labels.
        """
        raise NotImplementedError()

    def specs(self):
        """
        Returns the specifications of the ImageTargets array.

        Returns
        -------
        specs : str
            targets array format specifications.
        """
        raise NotImplementedError()

    def single(self, x):
        """
        Returns a single element from image targets.

        Parameters
        ----------
        x: int
            id of element to return

        Returns
        -------
        element: ImageTargets like
            ImageTargets with a single element

        Note
        ----
        The type of the single element will be the same,
        as for the initial container.
        """
        return self[x].reshape(1, -1)


class Keypoints(ImageTargets):
    """
    Subclass of ImageTargets designed to store the keypoints.
    This is the base class for keypoints arrays of different types.

    Parameters
    ----------
    input_array : ndarray
        the array to create Keypoints from.
        Must have float dtype and the shape (num_keypoints, param_num).
        If there is no_keypoints, it should have the shape (0,).


    Methods
    -------
    x_coords()
        returns the x coordinates of the keypoints as ints.
    y_coords()
        returns the x coordinates of the keypoints as ints.

    """

    def __init__(self, *args, **kwargs):
        self.param_num = 3
        self._check_shape()
        self._check_dtype()

    def x_coords(self):
        """
        Returns the x coordinates of the keypoints.

        Returns
        -------
        x_coords : ndarray of int
            1D array with x coordinates.
        """
        return self[:, 0].astype(int)

    def y_coords(self):
        """
        Returns the y coordinates of the keypoints

        Returns
        -------
        x_coords : ndarray of int
            1D array with y coordinates.
        """
        return self[:, 1].astype(int)

    def classes(self):
        """
        Returns keypoints classes

        Returns
        -------
        classes : ndarray of int
            1D array with y coordinates.
        """
        return self[:, 2].astype(int)

    def specs(self):
        """
        Returns spectifications of keypoints array
        """
        return "(x, y, class)"


# class ExtendedDataset(PointsDataset, Dataset):
#     def __getitem__(self, x):
#         super(ExtendedDataset, self).__getitem__(x)

#         image = load_image(self.images_paths[x])
#         keypoints = load_keypoints(self.labels_paths[x])

#         get_shifted_path = lambda path, postfix: path.split('.')[
#                                                      0] + postfix + '.' + path.split('.')[1]

#         image_path = self.images_paths[x]
#         if self.extended:
#             image_tl = load_image(get_shifted_path(image_path, '_TL'))
#             image_tr = load_image(get_shifted_path(image_path, '_TR'))
#             image_bl = load_image(get_shifted_path(image_path, '_BL'))
#             image_br = load_image(get_shifted_path(image_path, '_BR'))

#         class_labels = [x[-1] for x in keypoints]
#         keypoints_no_class = [x[:-1] for x in keypoints]

#         keypoints = [
#             np.array(y + (x,))
#             for x, y in zip(class_labels, keypoints_no_class)
#         ]

#         if keypoints:
#             keypoints = np.stack(keypoints)
#         else:
#             keypoints = np.empty((0, 3))

#         if self.extended:
#             to_return = {
#                 "keypoints": Keypoints(keypoints.astype(self.keypoints_dtype)),
#                 "image": image,
#                 "image_tl": image_tl,
#                 "image_tr": image_tr,
#                 "image_bl": image_bl,
#                 "image_br": image_br,
#             }
#         else:
#             to_return = {
#                 "keypoints": Keypoints(keypoints.astype(self.keypoints_dtype)),
#                 "image": image,
#             }

#         return to_return


class ImageTargetsBatch(ImageTargets):
    """
    Base class for targets for images batch.

    Methods
    -------
    image_labels()
        returns labels of images in the batch for each target
    from_image(image_i)
        returns the targets from a specific image

    Note
    ----
    The first number for each target must indicate image label, so the format should be like
    (image_label, x, y, class) or (image_label, x, y, class, confidence)

    See also
    --------
    pointdet.utils.targets.ImageTargets
        the base class with api specifications
    """

    def image_labels(self):
        """
        Returns labels of images in the batch for each target

        Returns
        -------
        image_labels : ndarray of int
            image labels for all targets
        """
        raise NotImplementedError()

    def from_image(self, image_i):
        """
        Returns the targets from a specific image.

        Parameters
        ----------
        image_i : int
            label of the image to take the targets from is. If not present, the excpetion will be raised.

        Returns
        -------
        targets : ImageTargets
            image labels for all keypoints.
        """

        raise NotImplementedError()

    def image_labels(self):
        """
        Returns image_labels
        """
        raise NotImplementedError()

    def num_images(self):
        """
        Returns number of images in a batch
        """

        raise NotImplementedError()


class KeypointsBatch(ImageTargetsBatch, Keypoints):
    """
    Base class for keypoints for image batch.

    Methods
    -------
    image_labels()
        returns labels of images in th batch for each keypoint
    from_image(image_i)
        returns the keypoints from a specific image

    Note
    ----
    The first number for each keypoit must indicate image label, so the format should be like
    (image_label, x, y, class) or (image_label, x, y, class, confidence)

    See also
    --------
    pointdet.utils.keypoints.Keypoints
        the base class with api specifications
    """

    def __init__(self, *args, **kwargs):
        self.param_num = 4
        self._check_shape()
        self._check_dtype()

    def _prepare_array_from_image(self, image_i):
        """
        Prepares the array from a given image

        Parameters
        ----------
        image_i : int
            label of the image to take the keypoints from is. If not present, the excpetion will be raised.

        Returns
        -------
        keypoints : ndarray
            image labels for all keypoints.
        """
        if self.shape[0] == 0:
            return self[:, 1:]
        mask = self.image_labels() == image_i

        if mask.sum() == 0:
            return np.empty((0, self.shape[1] - 1))
        #             raise Exception("No image with label %i" % int(image_i))
        return np.array(np.array(self[mask][:, 1:]))

    def image_labels(self):
        """
        Returns image_labels
        """
        return self[:, 0].astype(int)

    def num_images(self):
        """
        Returns number of images in a batch
        """

        return len(np.unique(self.image_labels()))

    def from_image(self, image_i):
        return Keypoints(self._prepare_array_from_image(image_i))

    def x_coords(self):
        return self[:, 1].astype(int)

    def y_coords(self):
        return self[:, 2].astype(int)

    def classes(self):
        return self[:, 3].astype(int)

    def specs(self):
        return "(image_i, x, y, class)"


def rescale_keypoints(keypoints, in_image_shape, out_image_shape):
    """
    Rescales keypoints coordinates.

    Parameters
    ----------
    keypoints: endoanalysis.targets.keypoints
        keypoints to rescale
    in_image_shape: tuple of int
        initial image shape, the format is (y_size, x_size)
    out_image_shape: tuple of int
        resulting image shape, the format is (y_size, x_size)

    Returns
    -------
    keypoints_rescaled: endoanalysis.targets.keypoints
        rescaled_keypoints
    """

    keypoints_rescaled = keypoints.copy()

    x_coords = keypoints_rescaled.x_coords().astype(float)
    y_coords = keypoints_rescaled.y_coords().astype(float)
    classes = keypoints_rescaled.classes().astype(float)

    x_coords = np.round((x_coords * out_image_shape[1] / in_image_shape[1]))
    y_coords = np.round((y_coords * out_image_shape[0] / in_image_shape[0]))

    if type(keypoints) is Keypoints:
        keypoints_rescaled = Keypoints(np.vstack([x_coords, y_coords, classes]).T)
    elif type(keypoints) is KeypointsBatch:
        image_lables = keypoints_rescaled.image_labels().astype(float)
        keypoints_rescaled = KeypointsBatch(
            np.vstack([image_lables, x_coords, y_coords, classes]).T
        )
    return keypoints_rescaled


def define_borders(image_x, image_y, window_x, window_y, kp_xs, kp_ys):
    x_l = kp_xs - (window_x - 1) / 2
    x_r = kp_xs + (window_x - 1) / 2
    y_l = kp_ys - (window_y - 1) / 2
    y_r = kp_ys + (window_y - 1) / 2

    x_min = np.max([x_l, np.zeros(len(kp_xs))], axis=0).astype(int)
    x_max = np.min([x_r, np.ones(len(kp_xs)) * image_x - 1], axis=0).astype(int) + 1
    y_min = np.max([y_l, np.zeros(len(kp_ys))], axis=0).astype(int)
    y_max = np.min([y_r, np.ones(len(kp_ys)) * image_y - 1], axis=0).astype(int) + 1

    window_x_min = np.where(x_l >= 0, 0, -x_l).astype(int)
    window_x_max = np.where(x_r < image_x, window_x, window_x - x_r + image_x - 1).astype(int)
    window_y_min = np.where(y_l >= 0, 0, -y_l).astype(int)
    window_y_max = np.where(y_r < image_y, window_y, window_y - y_r + image_y - 1).astype(int)

    return np.vstack([x_min, x_max, y_min, y_max]).T, np.vstack(
        [window_x_min, window_x_max, window_y_min, window_y_max]).T


def make_heatmap(x_size, y_size, keypoints, num_classes, base_bell):
    """
    Transforms keypoints to heatmap.

    Parameters
    ----------
    x_size: int
        x_size of the heatmap
    y_size: int
        y_size of heatmap
    keypoints: endoanalysis.targets.Keypoints
        keypoints to trasform
    sigma: int or ndarray of int
        standart deviation in pixels.
        If int, all keypoints will have the same sigmas,
        if ndarray, should have the shape (num_keypoits,)
    num_classes: int
        number of keypoints classes.

    Returns
    -------
    heatmap: ndarray
        heatmap for a given image. The shape is (num_classes, y_size, x_size)
    """

    bell_y, bell_x = base_bell.shape
    images_borders, windows_borders = define_borders(x_size, y_size, bell_x, bell_y, keypoints.x_coords(),
                                                     keypoints.y_coords())

    heatmaps = np.zeros((num_classes, y_size, x_size))
    classes = keypoints.classes().astype(int)
    for image_borders, window_borders, class_i in zip(images_borders, windows_borders, classes):
        x_min, x_max, y_min, y_max = image_borders
        window_x_min, window_x_max, window_y_min, window_y_max = window_borders
        current_slice = heatmaps[class_i, y_min:y_max, x_min:x_max]
        bell = base_bell[window_y_min:window_y_max, window_x_min:window_x_max]

        heatmaps[class_i, y_min:y_max, x_min:x_max] = np.maximum(current_slice, bell)

    return heatmaps


def keypoints_list_to_batch(keypoints_list):
    """
    Transforms list of keypoints to KeypointsBatch object

    Parameters
    ----------
    keypoints_list : list of Keypoints
        keypoints list to transform

    Returns
    -------
    batch : KeypointsBatch
        transformed batch
    """
    keypoints_return = []

    current_type = type(keypoints_list[0])

    # if current_type == Keypoints:
    #     batch_type = KeypointsBatch
    # else:
    #     raise Exception("Unsupported keypoints type %s" % current_type)
    batch_type = KeypointsBatch
    for image_i, keypoints in enumerate(keypoints_list):
        if len(keypoints) != 0:
            image_labels = (np.ones(keypoints.shape[0]) * image_i)[:, np.newaxis]
            keypoints = np.hstack([image_labels, keypoints])
            keypoints_return.append(keypoints)

    return batch_type(np.concatenate(keypoints_return, 0))


def collate_im_kp_hm(samples):
    images = [x["image"] for x in samples]
    keypoints_groups = [x["keypoints"] for x in samples]
    heatmaps = [x["heatmaps"] for x in samples]

    return_dict = {
        "image": torch.stack(images, 0).contiguous(),
        "keypoints": keypoints_list_to_batch(keypoints_groups),
        "heatmaps": torch.stack(heatmaps, 0).contiguous(),
    }

    return return_dict


class HeatmapsDataset(PointsDataset, Dataset):
    "Dataset with images, keypoints and the heatmaps corresdonding to them."

    def __init__(
            self,
            images_list,
            labels_list,
            class_labels_map={},
            model_in_channels=3,
            normalization=None,
            resize_to=None,
            sigma=1,
            augs_list=[],
            interpolation=cv2.INTER_LINEAR,
            heatmaps_shape=None,
            class_colors={0: (1, 0, 0), 1: (0, 1, 0), 2: (0, 0, 1)},
    ):

        super(HeatmapsDataset, self).__init__(
            images_list=images_list, labels_list=labels_list
        )

        self.sigma = sigma

        self.num_classes = len(set(class_labels_map.values()))
        self.class_labels_map = class_labels_map

        self.heatmaps_shape = heatmaps_shape
        self.normalization = normalization

        self.model_in_channels = model_in_channels

        if resize_to:
            self.resize_transform = A.augmentations.Resize(*resize_to, interpolation=interpolation)
            augs_list.append(self.resize_transform)
        self.augs_list = augs_list

        self.enable_augs()
        self.class_colors = class_colors
        self.heatmap_bell = self.create_heatmap_bell()

    def enable_augs(self):
        self.alb_transforms = A.Compose(
            self.augs_list,
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["class_labels"]
            ),
        )

    def disable_augs(self):
        self.alb_transforms = A.Compose(
            [self.resize_transform],
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["class_labels"]
            ),
        )

    def create_heatmap_bell(self):
        window_size = np.round(4 * self.sigma).astype(int)
        if window_size % 2 == 0:
            window_size += 1
        center_i = int((window_size - 1) / 2)

        ys, xs = np.indices((window_size, window_size))
        ys -= center_i
        xs -= center_i
        base_bell = np.exp(-(ys * ys + xs * xs) / (2 * self.sigma ** 2))
        return base_bell

    def __getitem__(self, x):

        sample = super(HeatmapsDataset, self).__getitem__(x)
        # sample["image_path"] = self.images_paths[x]
        # sample["labels_path"] = self.labels_paths[x]

        y_size, x_size, _ = sample["image"].shape

        if self.alb_transforms is not None:
            keypoints_no_class = np.stack(
                [sample["keypoints"].x_coords(), sample["keypoints"].y_coords()]
            ).T
            classes = list(sample["keypoints"].classes())

            transformed = self.alb_transforms(
                image=sample["image"],
                keypoints=keypoints_no_class,
                class_labels=classes,
            )

            kp_coords = np.array(transformed["keypoints"])
            classes = np.array(transformed["class_labels"]).reshape(-1, 1)

            sample["keypoints"] = Keypoints(
                np.hstack([kp_coords, classes]).astype(float)
            )
            sample["image"] = transformed["image"]

        if self.class_labels_map:
            labels_to_keep = self.class_labels_map.keys()
            kp_filtered = []
            for class_label in labels_to_keep:
                kp_class = sample["keypoints"][
                    sample["keypoints"].classes() == float(class_label)
                    ]
                new_classes = (
                        np.ones(len(kp_class)) * self.class_labels_map[class_label]
                )
                kp_class = Keypoints(
                    np.vstack([kp_class.x_coords(), kp_class.y_coords(), new_classes]).T
                )
                kp_filtered.append(kp_class)

            sample["keypoints"] = Keypoints(np.vstack(kp_filtered))

        if self.heatmaps_shape:
            keypoints_to_heatmap = rescale_keypoints(
                sample["keypoints"], sample["image"].shape, self.heatmaps_shape
            )
            y_size, x_size = self.heatmaps_shape
        else:
            keypoints_to_heatmap = sample["keypoints"]

        sample["heatmaps"] = make_heatmap(
            x_size, y_size, keypoints_to_heatmap, self.num_classes, self.heatmap_bell
        )

        sample["image"] = np.moveaxis(sample["image"], -1, 0)

        for key in ["heatmaps", "image"]:
            sample[key] = torch.tensor(sample[key]).float()

        if self.normalization:
            sample["image"] -= torch.tensor(self.normalization["mean"]).reshape(
                -1, 1, 1
            )
            sample["image"] /= torch.tensor(self.normalization["std"]).reshape(-1, 1, 1)

        if self.model_in_channels == 1:
            sample["image"] = sample["image"].mean(axis=0)[np.newaxis]

        return sample

    def collate(self, samples):
        return collate_im_kp_hm(samples)

    def visualize(
            self,
            x,
            show_labels=True,
            labels_kwargs={"radius": 3, "alpha": 1.0, "ec": (0, 0, 0)},
    ):

        sample = self[x]
        if self.normalization:
            sample["image"] = sample["image"] * torch.tensor(
                self.normalization["std"]
            ).view(-1, 1, 1) + torch.tensor(self.normalization["mean"]).view(-1, 1, 1)
        sample["image"] = sample["image"].int().numpy()
        sample["image"] = np.moveaxis(sample["image"], 0, -1)

        if show_labels:
            keypoints = sample["keypoints"]
        else:
            keypoints = Keypoints(np.empty((0, 3)))

        _ = visualize_keypoints(
            sample["image"],
            keypoints,
            class_colors=self.class_colors,
            circles_kwargs=labels_kwargs,
        )


def define_model(config):
    if (len(set(config["class_labels_map"].values())) !=
            config["model_kwargs"]["classes"]):
        raise ValueError(
            "Number of classes is not the same as the number of values in the class_labels_map."
        )

    model = globals()[config["model_name"]](**config["model_kwargs"])
    # if config.get("weights_path") is not None:
    #     model.load_state_dict(torch.load(config["weights_path"], map_location='cpu'))

    return model


def parse_master_yaml(yaml_path):
    """
    Imports master yaml and converts paths to make the usable from inside the script

    Parameters
    ----------
    yaml_path : str
        path to master yaml from the script

    Returns
    -------
    lists : dict of list of str
        dict with lists pf converted paths
    """
    with open(yaml_path, "r") as file:
        lists = yaml.safe_load(file)

    for list_type, paths_list in lists.items():
        new_paths_list = []
        for path in paths_list:
            new_path = os.path.join(os.path.dirname(yaml_path), path)
            new_path = os.path.normpath(new_path)
            new_paths_list.append(new_path)
        lists[list_type] = new_paths_list

    return lists


def define_extrapolation_mode(extrapolation_mode):
    if extrapolation_mode == "BORDER_CONSTANT":
        return cv2.BORDER_CONSTANT
    elif extrapolation_mode == "BORDER_REPLICATE":
        return cv2.BORDER_REPLICATE
    elif extrapolation_mode == "BORDER_WRAP":
        return cv2.BORDER_WRAP
    elif extrapolation_mode == "BORDER_REFLECT":
        return cv2.BORDER_REFLECT
    elif extrapolation_mode == "BORDER_REFLECT_101":
        return cv2.BORDER_REFLECT_101
    else:
        raise Exception(
            "\nExtrapolation mode should be one of the following: \n" +
            "BORDER_CONSTANT, BORDER_REPLICATE,BORDER_REFLECT, BORDER_WRAP, BORDER_REFLECT_101 \n"
            + "Got:\n%s" % extrapolation_mode)


def albumentations_from_config(aug_config):
    border_mode = define_extrapolation_mode(aug_config["border_mode"])
    augs_list = [
        A.augmentations.HueSaturationValue(
            p=aug_config["p_hsv"],
            hue_shift_limit=5,
            sat_shift_limit=5,
            val_shift_limit=5,
        ),
        A.augmentations.GaussNoise(p=aug_config["p_noise"],
                                   var_limit=aug_config["noise_var"]),
        A.augmentations.Rotate(
            limit=aug_config["rotate_angle"],
            p=aug_config["p_rotate"],
            border_mode=border_mode,
        ),
        A.augmentations.ShiftScaleRotate(
            shift_limit=aug_config["shift_factor"],
            scale_limit=0,
            rotate_limit=0,
            border_mode=border_mode,
            p=aug_config["p_shift"],
        ),
        A.augmentations.ShiftScaleRotate(
            shift_limit=0,
            scale_limit=aug_config["scale_factor"],
            rotate_limit=0,
            border_mode=border_mode,
            p=aug_config["p_scale"],
        ),
        A.augmentations.Perspective(
            scale=(0, aug_config["perspective_factor"]),
            p=aug_config["p_perspective"],
            interpolation=border_mode,
        ),
        A.augmentations.HorizontalFlip(p=aug_config["p_flip_hor"]),
        A.augmentations.VerticalFlip(p=aug_config["p_flip_vert"]),
    ]

    return augs_list


def get_heatmaps_dataset(master_yaml_path, config):
    master_yaml = parse_master_yaml(master_yaml_path)
    augs_list = albumentations_from_config(config["train"]["augmentations"])

    return HeatmapsDataset(
        master_yaml["images_lists"],
        master_yaml["labels_lists"],
        class_labels_map=config["model"]["class_labels_map"],
        sigma=config["data"]["heatmaps_sigma"],
        augs_list=augs_list,
        heatmaps_shape=config["data"]["image_shape"],
        normalization={
            "mean": config["data"]["norm_mean"],
            "std": config["data"]["norm_std"],
        },
        interpolation=config["train"]["interpolation"],
        model_in_channels=config["model"]["model_kwargs"]["in_channels"],
        resize_to=config["model"]["input_shape"],
    )


def parse_eval_config(config):
    eval_meters = None
    if config["metric"]["type"] == "mAP":
        similarity = KPSimilarity(scale=config["metric"]["similarity_scale"])
        eval_meters = mAPmetric(
            similarity,
            class_labels=config["metric"]["class_labels"],
            sim_thresh=config["metric"]["sim_threshold"],
            name_group="mAP",
            # name="_".join(["mAP"] + [str(x) for x in item["class_labels"]]),
            name="_".join([str(x) for x in config["metric"]["class_labels"]]),
        )
    else:
        raise Exception("Unknown meter type in eval: %s" %
                        config["metric"]["type"])

    return eval_meters


def remove_dummy_files():
    if os.path.exists('dummy/dummy_images.txt'):
        os.remove('dummy/dummy_images.txt')

    if os.path.exists('dummy/dummy_labels.txt'):
        os.remove('dummy/dummy_labels.txt')

    if os.path.exists('dummy/dummy.txt'):
        os.remove('dummy/dummy.txt')

    if os.path.exists('dummy/dummy.png'):
        os.remove('dummy/dummy.png')

    if os.path.isdir('dummy'):
        os.rmdir('dummy')


def make_dummy_labels_list(img_path):
    if os.path.exists('dummy/dummy_labels.txt'):
        os.remove('dummy/dummy_labels.txt')

    if os.path.exists('dummy/dummy_images.txt'):
        os.remove('dummy/dummy_images.txt')

    if os.path.exists('dummy/dummy.txt'):
        os.remove('dummy/dummy.txt')

    if os.path.exists('dummy/dummy.png'):
        os.remove('dummy/dummy.png')

    if not os.path.exists('dummy'):
        os.mkdir('dummy')

    shutil.copyfile(img_path, 'dummy/dummy.png')

    with open('dummy/dummy_images.txt', 'w') as f:
        f.write('dummy.png')

    with open('dummy/dummy_labels.txt', 'w') as f:
        f.write('dummy.txt')

    with open('dummy/dummy.txt', 'w') as f:
        f.write('1 1 0')


def make_dataset_from_image(img_path, config_path):
    """
    Creates dataset from image

    Args:
        img_path (str): Path to image
        config_path (str): Path to config file

    Returns:
        tuple:
            dataset (HeatmapsDataset): Dataset with image and dummy label
            config (dict): Config file
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    make_dummy_labels_list(img_path)

    dataset = HeatmapsDataset(
        ['dummy/dummy_images.txt'],
        ['dummy/dummy_labels.txt'],
        class_labels_map=config["model"]["class_labels_map"],
        sigma=config["data"]["heatmaps_sigma"],
        augs_list=[],
        heatmaps_shape=config["data"]["image_shape"],
        normalization={
            "mean": config["data"]["norm_mean"],
            "std": config["data"]["norm_std"],
        },
        interpolation=config["train"]["interpolation"],
        model_in_channels=config["model"]["model_kwargs"]["in_channels"],
        resize_to=config["model"]["input_shape"],
    )

    return (dataset, config)
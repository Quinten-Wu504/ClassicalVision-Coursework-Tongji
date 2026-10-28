import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt


class EdgeEvaluator:
    def __init__(self, max_dist=0.0075, fast_mode=False):
        """
        参数:
            max_dist (float): 匹配距离（相对于图像对角线）。
            fast_mode (bool): 是否启用快速评估，仅使用固定阈值（如128）评估。
        """
        self.max_dist = max_dist
        self.fast_mode = fast_mode

    def evaluate_image(self, edge_map, gt_edges, image_shape):
        """
        评估一张图像的边缘检测结果（多 threshold 扫描 or 快速评估）。

        参数:
            edge_map: 单通道边缘图 (0-255 uint8)。
            gt_edges: ground truth 二值图像（0或1）。
            image_shape: 原始图像形状。

        返回:
            dict: precision, recall, f1
        """
        edge_map = edge_map.astype(np.uint8)
        gt_edges = gt_edges.astype(np.uint8)

        diagonal = np.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)
        dist_threshold = self.max_dist * diagonal

        thresholds = [128] if self.fast_mode else np.linspace(0, 255, 20).astype(np.uint8)
        best_f1 = best_precision = best_recall = 0

        dt_gt = distance_transform_edt(1 - gt_edges)  # 距离为0的点为 edge，计算其余点到最近边缘的距离

        for thresh in thresholds:
            binary_edge = (edge_map >= thresh).astype(np.uint8)
            detected = np.sum(binary_edge)
            if detected == 0:
                continue

            match = (dt_gt <= dist_threshold) & (binary_edge == 1)
            matched = np.sum(match)

            gt_total = np.sum(gt_edges)
            if gt_total == 0:
                continue

            precision = matched / detected if detected > 0 else 0
            recall = matched / gt_total if gt_total > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1, best_precision, best_recall = f1, precision, recall

        return {'precision': best_precision, 'recall': best_recall, 'f1': best_f1}

    def evaluate_dataset(self, edge_maps, gt_edges_list, image_shapes):
        """
        评估整个数据集，支持 OIS/ODS 评估。

        参数:
            edge_maps (list): 每张图的边缘图。
            gt_edges_list (list): 每张图像的多个 ground truth 列表。
            image_shapes (list): 每张图像的形状。

        返回:
            dict: ODS 和 OIS 指标
        """
        ois_f1s, ois_precisions, ois_recalls = [], [], []

        for edge_map, gt_list, shape in tqdm(zip(edge_maps, gt_edges_list, image_shapes), total=len(edge_maps)):
            # 每张图像使用多个 GT，选择 F1 最高的结果
            best_result = {'f1': 0, 'precision': 0, 'recall': 0}
            for gt in gt_list:
                result = self.evaluate_image(edge_map, gt, shape)
                if result['f1'] > best_result['f1']:
                    best_result = result
            ois_f1s.append(best_result['f1'])
            ois_precisions.append(best_result['precision'])
            ois_recalls.append(best_result['recall'])

        ois_f1 = np.mean(ois_f1s)
        ois_precision = np.mean(ois_precisions)
        ois_recall = np.mean(ois_recalls)

        if self.fast_mode:
            # 快速模式下 ODS 不做多阈值评估
            ods_f1 = ois_f1
            ods_precision = ois_precision
            ods_recall = ois_recall
        else:
            thresholds = np.linspace(0, 255, 20).astype(np.uint8)
            best_avg_f1 = best_avg_precision = best_avg_recall = 0

            for thresh in thresholds:
                f1s, precisions, recalls = [], [], []
                for edge_map, gt_list, shape in zip(edge_maps, gt_edges_list, image_shapes):
                    bin_edge_map = (edge_map >= thresh).astype(np.uint8) * 255
                    for gt in gt_list:
                        result = self.evaluate_image(bin_edge_map, gt, shape)
                        f1s.append(result['f1'])
                        precisions.append(result['precision'])
                        recalls.append(result['recall'])
                avg_f1 = np.mean(f1s)
                if avg_f1 > best_avg_f1:
                    best_avg_f1 = avg_f1
                    best_avg_precision = np.mean(precisions)
                    best_avg_recall = np.mean(recalls)

            ods_f1 = best_avg_f1
            ods_precision = best_avg_precision
            ods_recall = best_avg_recall

        return {
            'ois_f1': ois_f1,
            'ois_precision': ois_precision,
            'ois_recall': ois_recall,
            'ods_f1': ods_f1,
            'ods_precision': ods_precision,
            'ods_recall': ods_recall
        }

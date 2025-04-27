import numpy as np
from sklearn.metrics import auc
from abc import ABC, abstractmethod
import utils

# Parameters for AD metrics
AD_LEVEL_PARAMS = [
    # AD1
    {
        'recall_alpha': 1.0,
        'recall_omega': 'default', 'recall_delta': 'flat', 'recall_gamma': 'dup',
        'precision_omega': 'default', 'precision_delta': 'flat', 'precision_gamma': 'dup'
    },
    # AD2
    {
        'recall_alpha': 0.0,
        'recall_omega': 'default', 'recall_delta': 'flat', 'recall_gamma': 'dup',
        'precision_omega': 'default', 'precision_delta': 'flat', 'precision_gamma': 'dup'
    },
    # AD3
    {
        'recall_alpha': 0.0,
        'recall_omega': 'flat.normalized', 'recall_delta': 'front', 'recall_gamma': 'dup',
        'precision_omega': 'default', 'precision_delta': 'flat', 'precision_gamma': 'dup'
    },
    # AD4
    {
        'recall_alpha': 0.0,
        'recall_omega': 'flat.normalized', 'recall_delta': 'front', 'recall_gamma': 'no.dup',
        'precision_omega': 'default', 'precision_delta': 'flat', 'precision_gamma': 'no.dup'
    }
]

class Evaluator(ABC):
    def __init__(self, beta):
        """
        Base class for anomaly detection evaluation.

        Args:
            beta (float): Weighting factor for recall in F_beta score.
        """
        self.beta = beta

    @abstractmethod
    def compute_binary_metrics(self, traces_labels, traces_preds):
        """
        Compute metrics for anomaly detection.

        Args:
            traces_labels (list): List of true labels for traces.
            traces_preds (list): List of predicted labels for traces.

        Returns:
            tuple: (F-scores, Precision, Recall)
        """
        pass

    @abstractmethod
    def compute_multiclass_metrics(self, traces_labels, traces_preds):
        """
        Compute metrics for anomaly detection.

        Args:
            traces_labels (list): List of true labels for traces.
            traces_preds (list): List of predicted labels for traces.

        Returns:
            tuple: (F-scores, Precision, Recall)
        """
        pass

class ADMetrics(Evaluator):
    def __init__(self, beta, num_classes, ad_level):
        """
        Extended evaluator for multiple anomaly types and ranges.

        Args:
            beta (float): Weighting factor for recall in F_beta score.
            num_classes (dict): Mapping of labels to indices.
            ad_level (int): AD metric level (1, 2, 3, or 4).
        """
        super().__init__(beta)
        self.num_classes = num_classes
        self.params = AD_LEVEL_PARAMS[ad_level - 1]

        # Recall specifications
        self.recall_alpha = self.params['recall_alpha']
        self.recall_omega = ADMetrics.omega_functions[self.params['recall_omega']]
        self.recall_delta = ADMetrics.delta_functions[self.params['recall_delta']]
        self.recall_gamma = ADMetrics.gamma_functions[self.params['recall_gamma']]

        # Precision specifications
        self.precision_omega = self.omega_functions[self.params['precision_omega']]
        self.precision_delta = self.delta_functions[self.params['precision_delta']]
        self.precision_gamma = self.gamma_functions[self.params['precision_gamma']]


    def compute_binary_metrics(self, traces_labels, traces_preds):
        """
        Compute metrics for binary anomaly types.

        Args:
            traces_labels (list): List of true anomaly labels.
            traces_preds (list): List of predicted anomaly labels.

        Returns:
            tuple: (F-scores, Precision, Recall, Type-wise metrics)
        """
        precisions = []
        recalls = [] 
        type_wise_metrics = {}
        for anomaly_type in range(1, self.num_classes):
            type_wise_metrics[anomaly_type] = {"recall": 0, "f_score": 0, "recalls": []}
       
        for trace_labels, trace_preds in zip(traces_labels, traces_preds):
            true_anomalies = utils.extract_anomalies_from_trace(trace_labels)
            pred_anomalies = utils.extract_anomalies_from_trace(trace_preds)
            
            all_pred_ranges = [
                    interval for t, interval in zip(pred_anomalies["anomaly_type"], pred_anomalies["anomaly_intervals"]) 
                ]
            
            all_true_ranges = [
                    interval for t, interval in zip(true_anomalies["anomaly_type"], true_anomalies["anomaly_intervals"])
                ]
            
            if all_true_ranges:
                recall_ = sum(self.compute_range_recall(true_range, all_pred_ranges) for true_range in all_true_ranges) / len(all_true_ranges)
                recalls.append(recall_)

            if all_pred_ranges:
                precision_ = sum(self.compute_range_precision(pred_range, all_true_ranges) for pred_range in all_pred_ranges) / len(all_pred_ranges)
                precisions.append(precision_)

            for anomaly_type in range(1, self.num_classes):
                
                true_ranges = [
                    interval for t, interval in zip(true_anomalies["anomaly_type"], true_anomalies["anomaly_intervals"]) if t == anomaly_type
                ]

                # Compute recall for each true range
                if true_ranges:
                    recall_ = sum(self.compute_range_recall(true_range, all_pred_ranges) for true_range in true_ranges) / len(true_ranges)
                    type_wise_metrics[anomaly_type]["recalls"].append(recall_)
        
        # Average precision and recall
        precision = np.mean(precisions) if precisions else 0
        recall = np.mean(recalls) if recalls else 0
        f_score = self.get_f_beta_score(precision, recall)

        for anomaly_type, metrics in type_wise_metrics.items():     
            type_wise_metrics[anomaly_type]["recall"] = np.mean(type_wise_metrics[anomaly_type]["recalls"]) if type_wise_metrics[anomaly_type]["recalls"] else 0
            type_wise_metrics[anomaly_type]["f_score"] = self.get_f_beta_score(precision, type_wise_metrics[anomaly_type]["recall"])

        return f_score, precision, recall, type_wise_metrics

    def compute_multiclass_metrics(self, traces_labels, traces_preds):
        """
        Compute metrics for multiple anomaly types.

        Args:
            traces_labels (list): List of true anomaly labels.
            traces_preds (list): List of predicted anomaly labels.

        Returns:
            tuple: (F-scores, Precision, Recall, Type-wise metrics)
        """
        precisions = []
        recalls = [] 
        type_wise_metrics = {}
        for anomaly_type in range(1, self.num_classes):
            type_wise_metrics[anomaly_type] = {"precision": 0, "recall": 0, "f_score": 0, "precisions": [], "recalls": []}
       
        
        for trace_labels, trace_preds in zip(traces_labels, traces_preds):
            true_anomalies = utils.extract_anomalies_from_trace(trace_labels)
            pred_anomalies = utils.extract_anomalies_from_trace(trace_preds)

            for anomaly_type in (label for label in range(1, self.num_classes)):
                
                true_ranges = [
                    interval for t, interval in zip(true_anomalies["anomaly_type"], true_anomalies["anomaly_intervals"]) if t == anomaly_type
                ]
                pred_ranges = [
                    interval for t, interval in zip(pred_anomalies["anomaly_type"], pred_anomalies["anomaly_intervals"]) if t == anomaly_type
                ]

                # Compute recall for each true range
                if true_ranges:
                    recall_ = sum(self.compute_range_recall(true_range, pred_ranges) for true_range in true_ranges) / len(true_ranges)
                    recalls.append(recall_)
                    type_wise_metrics[anomaly_type]["recalls"].append(recall_)

                # Compute precision for each predicted range
                if pred_ranges:
                    precision_ = sum(self.compute_range_precision(pred_range, true_ranges) for pred_range in pred_ranges) / len(pred_ranges)
                    precisions.append(precision_)
                    type_wise_metrics[anomaly_type]["precisions"].append(precision_)
        
        # Average precision and recall
        precision = np.mean(precisions) if precisions else 0
        recall = np.mean(recalls) if recalls else 0
        f_score = self.get_f_beta_score(precision, recall)

        for anomaly_type, metrics in type_wise_metrics.items():     
            type_wise_metrics[anomaly_type]["precision"]  = np.mean(type_wise_metrics[anomaly_type]["precisions"]) if type_wise_metrics[anomaly_type]["precisions"] else 0
            type_wise_metrics[anomaly_type]["recall"] = np.mean(type_wise_metrics[anomaly_type]["recalls"]) if type_wise_metrics[anomaly_type]["recalls"] else 0
            type_wise_metrics[anomaly_type]["f_score"] = self.get_f_beta_score(type_wise_metrics[anomaly_type]["precision"], type_wise_metrics[anomaly_type]["recall"])

        return f_score, precision, recall, type_wise_metrics

    def compute_range_recall(self, true_range, pred_ranges):
        """
        Compute recall for a true range.

        Args:
            true_range (tuple): Start and end indices of the true range.
            pred_ranges (list): List of predicted ranges.

        Returns:
            float: Recall score.
        """

        return self.recall_alpha * ADMetrics.existence_reward(true_range, pred_ranges) + \
            (1 - self.recall_alpha) * ADMetrics.overlap_reward(true_range, pred_ranges, self.recall_omega, self.recall_delta, self.recall_gamma)

    def compute_range_precision(self, pred_range, true_ranges):
        """
        Compute precision for a predicted range.

        Args:
            pred_range (tuple): Start and end indices of the predicted range.
            true_ranges (list): List of true ranges.

        Returns:
            float: Precision score.
        """

        return ADMetrics.overlap_reward(pred_range, true_ranges, self.precision_omega, self.precision_delta, self.precision_gamma)
        

    @staticmethod
    def overlap(range1, range2):
        """
        Check if two ranges overlap.

        Args:
            range1 (tuple): First range (start, end).
            range2 (tuple): Second range (start, end).

        Returns:
            bool: True if ranges overlap, else False.
        """
        return not (range1[1] < range2[0] or range1[0] > range2[1])

    @staticmethod
    def existence_reward(range_, other_ranges):
        """
        Returns the existence reward of `range_` with respect to `other_ranges`.

        Args:
            range_ (tuple): Start and end indices of the range whose existence reward to compute.
            other_ranges (list): List of other ranges to test overlapping with.

        Returns:
            int: 1 if `range_` overlaps with at least one record of `other_ranges`, 0 otherwise.
        """
        return any(ADMetrics.overlap(range_, r) for r in other_ranges)

    @staticmethod
    def overlap_reward(range_, other_ranges, omega_f, delta_f, gamma_f):
        """Returns the overlap reward of `range_` with respect to `other_ranges` and
            the provided functions.

        Args:
            range_ (ndarray): start and end indices of the range whose overlap reward to compute.
            other_ranges (ndarray): 2d-array for the start and end indices of the "target" ranges.
            omega_f (func): size function.
            delta_f (func): positional bias.
            gamma_f (func): cardinality function.

        Returns:
            float: the overlap reward of `range_`, between 0 and 1.
        """
        size_rewards = 0
        for other_range in other_ranges:
            size_rewards += omega_f(range_, ADMetrics.get_overlapping_range(range_, other_range), delta_f)
        return ADMetrics.cardinality_factor(range_, other_ranges, gamma_f) * size_rewards

    @staticmethod
    def get_overlapping_range(range_1, range_2):
        """
        Returns the start and end indices of the overlap between `range_1` and `range_2`.

        If `range_1` and `range_2` do not overlap, None is returned.

        Args:
            range_1 (tuple): (start, end) indices of the first range.
            range_2 (tuple): (start, end) indices of the second range.

        Returns:
            tuple|None: (start, end) indices of the overlap, or None if no overlap exists.
        """
        overlap_start = max(range_1[0], range_2[0])
        overlap_end = min(range_1[1], range_2[1])
        if overlap_start <= overlap_end:
            return overlap_start, overlap_end
        return None

    @staticmethod
    def get_overlapping_ranges(target_range, ranges):
        """
        Returns the start and end indices of all overlapping ranges between `target_range` and `ranges`.

        If none of the ranges overlap with `target_range`, an empty list is returned.

        Args:
            target_range (tuple): Target range to overlap as (start, end).
            ranges (list): List of candidate ranges as [(start_1, end_1), (start_2, end_2), ...].

        Returns:
            list: List of overlapping ranges as [(start_1, end_1), ...].
        """
        overlaps = []
        for r in ranges:
            overlap = ADMetrics.get_overlapping_range(target_range, r)
            if overlap is not None:
                overlaps.append(overlap)
        return overlaps

    @staticmethod
    def cardinality_factor(true_range, pred_ranges, gamma_f):
        """Returns the cardinality factor of `range_` with respect to `other_ranges` and `gamma_f`.

        Args:
            true_range (ndarray): start and end indices of the range whose cardinality factor to compute.
            pred_ranges (ndarray): 2d-array for the start and end indices of the "target" ranges.
            gamma_f (func): cardinality function.

        Returns:
            float: the cardinality factor of `range_`, between 0 and 1.
        """
        n_overlapping_ranges = len(ADMetrics.get_overlapping_ranges(true_range, pred_ranges))
        if n_overlapping_ranges == 1:
            return 1
        return gamma_f(n_overlapping_ranges)
    
    def get_f_beta_score(self, precision, recall):
        """
        Compute F_beta score.

        Args:
            precision (float): Precision score.
            recall (float): Recall score.

        Returns:
            float: F_beta score.
        """
        if precision == recall == 0:
            return 0
        beta_squared = self.beta ** 2
        return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)
    
    """Omega (size) functions.
    
    Return the size reward of the overlap based on the positional bias of the target range.
    """
    @staticmethod
    def default_size_function(range_, overlap, delta_f):
        """Returns the reward as the overlap size weighted by the positional bias."""
        if overlap is None:
            return 0
        # Extract overlap start and end indices
        overlap_start, overlap_end = overlap
        # Normalize overlap indices relative to `range_`
        relative_start = overlap_start - range_[0]
        relative_end = overlap_end - range_[0] + 1
        # Compute positional bias rewards for the range
        range_rewards = delta_f(range_[1] - range_[0] + 1)
        # Return the total normalized reward covered by the overlap
        return sum(range_rewards[relative_start:relative_end])

    @staticmethod
    def flat_normalized_size_function(range_, overlap, delta_f):
        """Returns the overlap reward normalized so as not to exceed what it would be under a flat bias."""
        if overlap is None:
            return 0
        # Extract overlap start and end indices
        overlap_start, overlap_end = overlap
        # Normalize overlap indices relative to `range_`
        relative_start = overlap_start - range_[0]
        relative_end = overlap_end - range_[0] + 1
        # Compute rewards with the provided positional bias
        range_rewards = delta_f(range_[1] - range_[0] + 1)
        # Compute rewards with a flat bias
        flat_rewards = ADMetrics.delta_functions['flat'](range_[1] - range_[0] + 1)
        # Calculate total rewards covered by the overlap for both biases
        original_reward = sum(range_rewards[relative_start:relative_end])
        flat_reward = sum(flat_rewards[relative_start:relative_end])
        # Best achievable reward for the overlap size
        overlap_size = relative_end - relative_start
        max_reward = sum(sorted(range_rewards, reverse=True)[:overlap_size])
        # Normalize the original reward so that the maximum is the flat reward
        return (flat_reward * original_reward / max_reward) if max_reward != 0 else 0



    # dictionary gathering references to the defined `omega` size functions
    omega_functions = {
        'default': default_size_function.__func__,
        'flat.normalized': flat_normalized_size_function.__func__
    }

    """Delta functions (positional biases). 
    
    Return the normalized rewards for each relative index in a range of length `range_length`.
    """
    @staticmethod
    def flat_bias(range_length): 
        if range_length <= 0:
            return np.array([])
        return np.ones(range_length) / range_length

    @staticmethod
    def front_end_bias(range_length):
        """The index rewards linearly decrease as we move forward in the range.
        """
        if range_length <= 0:
            return np.array([])
        raw_rewards = np.flip(np.array(range(range_length)))
        return raw_rewards / sum(raw_rewards)

    @staticmethod
    def back_end_bias(range_length):
        """The index rewards linearly increase as we move forward in the range.
        """
        if range_length <= 0:
            return np.array([])
        raw_rewards = np.array(range(range_length))
        return raw_rewards / sum(raw_rewards)

    # dictionary gathering references to the defined `delta` positional biases
    delta_functions = {
        'flat': flat_bias.__func__,
        'front': front_end_bias.__func__,
        'back': back_end_bias.__func__
    }


    """Gamma functions (cardinality)
    """
    @staticmethod
    def no_duplicates_cardinality(n_overlapping_ranges): return 0

    @staticmethod
    def allow_duplicates_cardinality(n_overlapping_ranges): return 1

    @staticmethod
    def inverse_polynomial_cardinality(n_overlapping_ranges): return 1 / n_overlapping_ranges
    # dictionary gathering references to the defined `gamma` cardinality functions
    gamma_functions = {
        'no.dup': no_duplicates_cardinality.__func__,
        'dup': allow_duplicates_cardinality.__func__,
        'inv.poly': inverse_polynomial_cardinality.__func__
    }

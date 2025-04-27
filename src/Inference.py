import torch
import utils
import numpy as np
from collections import Counter
from collections import defaultdict

class BaseInference:
    def __init__(self, model, device, window_size, stride, num_future_samples, num_classes):
        """
        Base inference class.

        Args:
            model (nn.Module): The trained model to evaluate.
            device (torch.device): Device to perform evaluation on.
            window_size (int): the size of window
            stride (int): Step size for sliding windows.
            num_future_samples (int): Number of future predictions to evaluate.
            labels_vocab (dict): Mapping of labels to indices.
            num_classes (int): Size of the label vocabulary.
        """
        self.model = model
        self.device = device
        self.window_size = window_size
        self.stride = stride
        self.num_future_samples = num_future_samples
        self.num_classes = num_classes
        
    def forecast(self, traces_windows, traces_labels, allowed_transitions):
        raise NotImplementedError("Forecasting method should be implemented in the derived classes.")
    
    def apply_transition_rules(self, logits, prev_label, allowed_transitions):
        """
        Applies a mask to the logits based on allowed transitions.

        Args:
            logits (torch.Tensor): Logits from the model for a single time step.
            prev_label (int): The previous label.
            allowed_transitions (dict): Dictionary of allowed transitions.

        Returns:
            torch.Tensor: Masked logits.
        """
        allowed = allowed_transitions.get(prev_label, [])
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for a in allowed:
            mask[a] = True
        return logits.masked_fill(~mask, float('-inf'))
    

    def majority_first(self, preds):
        counts = Counter(preds)
        max_count = max(counts.values())
        candidates = [label for label, count in counts.items() if count == max_count]
    
        # Find the first label among candidates
        for i, label in enumerate(preds):
            if label in candidates:
                return label, i

    def majority_latest(self, preds, majority_window_size):
        
        # Trim to the last 'majority_window_size' elements if needed
        if len(preds) > majority_window_size:
            window_preds = preds[-majority_window_size:]
            offset = len(preds) - majority_window_size
        else:
            window_preds = preds
            offset = 0

        # Count frequencies
        counts = Counter(window_preds)
        max_count = max(counts.values())
        candidates = [label for label, count in counts.items() if count == max_count]

        # Find the latest occurrence of a candidate in the window
        for i in reversed(range(len(window_preds))):
            if window_preds[i] in candidates:
                return window_preds[i], i + offset  # Adjust index to original 'preds'

    
class AutoregressiveForecaster(BaseInference):
    def __init__(self, model, device, window_size, stride, num_future_samples, num_classes):
        super().__init__(model, device, window_size, stride, num_future_samples, num_classes)

    def forecast(self, traces_windows, traces_labels, allowed_transitions, mode, majority_window_size=10):
        """
        Evaluates the model on the provided dataset.

        Args:
            traces_windows (list): List of input time-series windows for traces.
            traces_labels (list): List of labels corresponding to the windows for traces.
            allowed_transitions (dict): Dictionary of allowed transitions.
            mode:
            (1): use latest predictions without masking
            (2): use majority labels without masking
            (3): use latest predictions with masking
            (4): use majority labels with masking

        Returns:
            tuple: (all_true_labels, all_predicted_labels)
        """
        self.model.eval()
        all_predicted_labels = []
        all_confidences = [] 
        all_probs = []

        with torch.no_grad():
            trace_index = -1
            for trace_windows, trace_labels in zip(traces_windows, traces_labels):
                trace_index += 1
                trace_predicted_labels = []
                trace_confidences = []
                trace_probs = []
                
                if (mode==2) or (mode==4):
                    trace_prediction_dict = defaultdict(list)
                    trace_confidence_dict = defaultdict(list)
                    trace_probs_dict = defaultdict(list)
       
                num_windows = len(trace_windows)
                for window_index, (window, window_labels) in enumerate(zip(trace_windows, trace_labels)):

                    if window_index >= num_windows - self.num_future_samples:
                        break

                    # Prepare model input for current window
                    if window_index != 0:
                        if window_index <= self.window_size:
                            start_index_of_preds = 0
                        else:
                            start_index_of_preds += 1
                        prev_labels = window_labels[:-window_index]
                        labels = prev_labels + trace_predicted_labels[start_index_of_preds:]
  
                    else:
                        labels = window_labels
    
                    labels_one_hot = np.eye(self.num_classes)[labels]
                    augmented_window = np.concatenate([window, labels_one_hot], axis=-1)
                    augmented_window_tensor = torch.tensor(augmented_window, dtype=torch.float32, device=self.device).unsqueeze(0)
                    # Forward pass for current window
                    outputs = self.model(augmented_window_tensor)

                    if (mode==1) or (mode==2):
                        predicted_labels = torch.argmax(outputs, dim=-1).squeeze(0).tolist()  # Shape: (num_future_samples,)
                        probabilities = torch.softmax(outputs, dim=-1)  # Convert logits to probabilities
                        confidences = torch.max(probabilities, dim=-1).values.squeeze(0).tolist()
                        probabilities = probabilities[0].tolist()

                    # Masked evaluation
                    if (mode==3) or (mode==4):
                        logits = outputs.squeeze(0)  # Shape: (num_future_samples, num_classes)
                        predicted_labels = []
                        probabilities = []
                        confidences = []
                        prev_label = labels[-1]
                        for t in range(self.num_future_samples):
                            step_logits = logits[t]
                            masked_logits = self.apply_transition_rules(step_logits, prev_label, allowed_transitions)
                            masked_pred_label = torch.argmax(masked_logits).item()
                            predicted_labels.append(masked_pred_label)
                            probabilities.append(torch.softmax(masked_logits, dim=0).tolist())  # Convert logits to probabilities
                            confidences.append(torch.max(torch.softmax(masked_logits, dim=0)).item())
                            prev_label = masked_pred_label  # Update for the next time step
                
                    # Majority selection
                    if (mode==2) or (mode==4):
                        majority_predicted_labels = []
                        majority_confidences = []
                        majority_probabilities = []
                        for t, (l, c, p) in enumerate(zip(predicted_labels, confidences, probabilities)):
                            trace_prediction_dict[window_index + t].append(l)
                            trace_confidence_dict[window_index + t].append(c)
                            trace_probs_dict[window_index + t].append(p)
                            majority_label, majority_index = self.majority_latest(trace_prediction_dict[window_index + t], majority_window_size)
                            majority_predicted_labels.append(majority_label)
                            majority_confidences.append(trace_confidence_dict[window_index + t][majority_index])
                            majority_probabilities.append(trace_probs_dict[window_index + t][majority_index])
       
                    # Add results for current window to a trace
                    if window_index == num_windows - self.num_future_samples - 1:
                        if (mode==1) or (mode==3):
                            trace_predicted_labels.extend(predicted_labels)
                            trace_confidences.extend(confidences)
                            trace_probs.extend(probabilities)
                        else:
                            trace_predicted_labels.extend(majority_predicted_labels)
                            trace_confidences.extend(majority_confidences)
                            trace_probs.extend(majority_probabilities)

                    else:
                        if (mode==1) or (mode==3):
                            trace_predicted_labels.extend(predicted_labels[:self.stride])
                            trace_confidences.extend(confidences[:self.stride])
                            trace_probs.extend((probabilities[:self.stride])) 
                        else:
                            trace_predicted_labels.extend(majority_predicted_labels[:self.stride])
                            trace_confidences.extend(majority_confidences[:self.stride])
                            trace_probs.extend((majority_probabilities[:self.stride])) 
                            
                # Store true and predicted labels for the entire trace
                all_predicted_labels.append(trace_predicted_labels)
                all_confidences.append(trace_confidences)
                all_probs.append(trace_probs)


        return all_predicted_labels, all_confidences, all_probs


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

class Experiment:
    def __init__(self):
        """
        Initializes an empty list to store SignalDetection objects and their labels.
        """
        self.conditions = []
        self.labels = []
    
    def add_condition(self, sdt_obj, label=None):
        """
        Adds a SignalDetection object and an optional label to the experiment.
        """
        if not isinstance(label, (str, type(None))):
            raise TypeError("Label must be a string or None.")
        self.conditions.append(sdt_obj)
        self.labels.append(label)
    
    def sorted_roc_points(self):
        """
        Returns sorted false alarm rates and hit rates for plotting the ROC curve.
        """
        if not self.conditions:
            raise ValueError("No conditions added to the experiment.")
        
        false_alarm_rates = [sdt.false_alarm_rate() for sdt in self.conditions]
        hit_rates = [sdt.hit_rate() for sdt in self.conditions]
        
        sorted_indices = np.argsort(false_alarm_rates)
        sorted_false_alarm_rates = np.array(false_alarm_rates)[sorted_indices].tolist()
        sorted_hit_rates = np.array(hit_rates)[sorted_indices].tolist()
        
        return sorted_false_alarm_rates, sorted_hit_rates
    
    def compute_auc(self):
        """
        Computes the Area Under the Curve (AUC) using the Trapezoidal Rule.
        """
        if not self.conditions:
            raise ValueError("No conditions added to the experiment.")
        
        false_alarm_rates, hit_rates = self.sorted_roc_points()
        return trapezoid(hit_rates, false_alarm_rates)
    
    def plot_roc_curve(self, show_plot=True):
        """
        Plots the ROC curve.
        """
        if not self.conditions:
            raise ValueError("No conditions added to the experiment.")
        
        false_alarm_rates, hit_rates = self.sorted_roc_points()
        
        plt.figure(figsize=(6, 6))
        plt.plot(false_alarm_rates, hit_rates, marker='o', linestyle='-', label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Chance Level')
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()
        
        if show_plot:
            plt.show()

# Example

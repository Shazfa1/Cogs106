import numpy as np
from scipy.stats import norm

class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections

    def d_prime(self):
        hit_rate = self.hits / (self.hits + self.misses)
        fa_rate = self.falseAlarms / (self.falseAlarms + self.correctRejections)
        
        # Adjust rates if they are 0 or 1
        hit_rate = max(0.01, min(hit_rate, 0.99))
        fa_rate = max(0.01, min(fa_rate, 0.99))
        
        z_hit = norm.ppf(hit_rate)
        z_fa = norm.ppf(fa_rate)
        
        return z_hit - z_fa

    def criterion(self):
        hit_rate = self.hits / (self.hits + self.misses)
        fa_rate = self.falseAlarms / (self.falseAlarms + self.correctRejections)
        
        # Adjust rates if they are 0 or 1
        hit_rate = max(0.01, min(hit_rate, 0.99))
        fa_rate = max(0.01, min(fa_rate, 0.99))
        
        z_hit = norm.ppf(hit_rate)
        z_fa = norm.ppf(fa_rate)
        
        return -0.5 * (z_hit + z_fa)

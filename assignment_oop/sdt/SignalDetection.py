import math
from scipy import stats

class SignalDetection(hits, misses, falseAlarms, correctRejections):
    def __init__(self):
        pass

    def hit_rate(self, hits, misses):
        """
        Calculate the hit rate.

        :param hits: Number of correctly detected signals
        :param misses: Number of signals that were not detected
        :return: Hit rate
        """
        return hits / (hits + misses)

    def false_alarm_rate(self, false_alarms, correct_rejections):
        """
        Calculate the false alarm rate.

        :param false_alarms: Number of noise trials incorrectly detected as signals
        :param correct_rejections: Number of noise trials correctly identified as noise
        :return: False alarm rate
        """
        return false_alarms / (false_alarms + correct_rejections)

    def d_prime(self, hit_rate, false_alarm_rate):
        """
        Calculate d-prime given hit rate and false alarm rate.
        
        :param hit_rate: The proportion of hits (between 0 and 1)
        :param false_alarm_rate: The proportion of false alarms (between 0 and 1)
        :return: The d-prime value
        """
        # Ensure the rates are not 0 or 1 to avoid infinite z-scores
        hit_rate = max(min(hit_rate, 0.99999), 0.00001)
        false_alarm_rate = max(min(false_alarm_rate, 0.99999), 0.00001)

        # Calculate z-scores
        z_hit = stats.norm.ppf(hit_rate)
        z_fa = stats.norm.ppf(false_alarm_rate)

        # Calculate d-prime
        return z_hit - z_fa

    def criterion(self, hit_rate, false_alarm_rate):
        """
        Calculate criterion given hit rate and false alarm rate.
        
        :param hit_rate: The proportion of hits (between 0 and 1)
        :param false_alarm_rate: The proportion of false alarms (between 0 and 1)
        :return: The criterion value
        """
        # Ensure the rates are not 0 or 1 to avoid infinite z-scores
        hit_rate = max(min(hit_rate, 0.99999), 0.00001)
        false_alarm_rate = max(min(false_alarm_rate, 0.99999), 0.00001)

        # Calculate z-scores
        z_hit = stats.norm.ppf(hit_rate)
        z_fa = stats.norm.ppf(false_alarm_rate)

        # Calculate criterion
        return -0.5 * (z_hit + z_fa)

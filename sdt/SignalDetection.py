mport math
from scipy import stats

class SignalDetectionTheory:
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
        :

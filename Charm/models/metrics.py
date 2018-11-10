class Metrics(object):
    @staticmethod
    def h_metric(design_candidate):
        cores = [d * 1. for d in design_candidate if d > 0]
        core_types = len(cores)
        core_counts = sum(cores) * 1.
        return core_types / core_counts

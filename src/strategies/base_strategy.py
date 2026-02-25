class BaseStrategy:
    def __init__(self, name="BaseStrategy"):
        self.name = name

    def generate_signals(self, df):
        raise NotImplementedError

    def position_size(self, df):
        return 1.0  # default fixed size

    def run(self, df):
        signals = self.generate_signals(df)
        return signals * self.position_size(df)

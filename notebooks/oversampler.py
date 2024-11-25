import pandas as pd
from imblearn.over_sampling import RandomOverSampler

class Oversampler:
    def __init__(self, engine='pyarrow'):
        self.engine = engine  # The engine to use (pyarrow or fastparquet)

    def oversample(self, df):
        x = df.drop(columns=['target'])  # All columns except 'y'
        y = df['target']  # Target column
        oversampler = RandomOverSampler(sampling_strategy='not majority', random_state=42)
        x_resampled, y_resampled = oversampler.fit_resample(x, y)
        df_resampled = pd.concat([pd.DataFrame(x_resampled, columns=x.columns), 
                          pd.Series(y_resampled, name='y')], axis=1)
        return df_resampled

import pandas as pd
from imblearn.over_sampling import RandomOverSampler

def oversample_dataframe(df):
    # Separate features and target
    x = df.drop(columns=["target"])  # All columns except target
    y = df["target"]  # Target column

    # Initialize the oversampler
    oversampler = RandomOverSampler(sampling_strategy="not majority", random_state=42)
    
    # Apply oversampling
    x_resampled, y_resampled = oversampler.fit_resample(x, y)
    
    # Combine oversampled features and target back into a DataFrame
    df_resampled = pd.concat([pd.DataFrame(x_resampled, columns=x.columns), 
                              pd.Series(y_resampled, name="target")], axis=1)
    
    return df_resampled


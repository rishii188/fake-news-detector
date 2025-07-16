import pandas as pd

# Load each dataset
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add labels
fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

# Combine into one dataframe
df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle the rows
df = df.sample(frac=1).reset_index(drop=True)

# Save the combined dataframe to a new CSV file
df.to_csv("combined_news_data.csv", index=False)

# Display the first few rows of the dataframe
print(df.head())

# Display the distribution of labels
print(df['label'].value_counts())




import os
import sys
import pandas as pd
import zipfile

from sklearn.preprocessing import LabelEncoder


def parse_dataframe(path: str) -> tuple[pd.DataFrame, dict]:
    folder = path

    labels_path = os.path.join(folder, "labels.json")
    csv_path = os.path.join(folder, "weatherHistory.csv")
    zip_path = os.path.join(folder, "weatherHistory.csv.zip")

    if os.path.exists(csv_path):
        os.remove(csv_path)

    if os.path.exists(labels_path):
        os.remove(labels_path)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(folder)

        print(f"Successfully extracted '{zip_path}' to '{csv_path}'.\n")

    except zipfile.BadZipFile:
        print(f"Error: '{zip_path}' is not a valid zip file.")
        sys.exit(1)

    except FileNotFoundError:
        print(f"Error: Zip file '{zip_path}' not found.")
        sys.exit(1)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    print("Parsing dataframe...\n")

    df = pd.read_csv(csv_path)

    # Remove redundant columns if they exist.
    df.drop(
        columns=["Formatted Date", "Daily Summary", "Loud Cover"],
        errors="ignore",
        inplace=True,
    )

    # Remove rows containing invalid values before encoding.
    for column in df.columns:
        invalid_rows = df[df[column].isin(["null", None])].index
        df.drop(index=invalid_rows, inplace=True)

    df.reset_index(drop=True, inplace=True)

    # Encode weather-summary labels and preserve their original names.
    summary_encoder = LabelEncoder()
    original_summaries = df["Summary"].astype(str)
    df["Summary"] = summary_encoder.fit_transform(original_summaries)

    labels = {
        str(index): str(label)
        for index, label in enumerate(summary_encoder.classes_)
    }

    # Encode precipitation type.
    precip_encoder = LabelEncoder()
    df["Precip Type"] = precip_encoder.fit_transform(
        df["Precip Type"].fillna("unknown").astype(str)
    )

    df.to_csv(csv_path, index=False)

    print("New fixed dataframe:")
    print(df.head())

    return df, labels

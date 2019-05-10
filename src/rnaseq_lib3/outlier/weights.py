import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import pca

sns.set_style("whitegrid")


class Weights:
    def __init__(self, tumor_path: pd.DataFrame, sample_dir: str):
        self.tumor_path = tumor_path
        self.tumor = self._load_tumor()
        self.genes = self.tumor.columns[5:]
        self.sample_dir = sample_dir
        self.df = self._weight_df()
        self.perc = self._perc_df()
        self.num_samples = len(self.df["sample"].unique())

    def _weight_df(self) -> pd.DataFrame:
        """
        Creates DataFrame of sample weights from a directory of samples
        Columns: tissue, normal_tissue, weight, sample_id

        Returns:
            DataFrame of Weights
        """
        # DataFrame: cols=tissue, normal_tissue, weight
        weights = []
        tissues = self.tumor.tissue
        for sample in os.listdir(self.sample_dir):
            sample_tissue = tissues.loc[sample]
            w = pd.read_csv(
                os.path.join(self.sample_dir, sample, "weights.tsv"), sep="\t"
            )
            w.columns = ["normal_tissue", "Median", "std"]
            w["tissue"] = sample_tissue
            w["sample"] = sample
            weights.append(w.drop("std", axis=1))
        return pd.concat(weights).reset_index(drop=True)

    def _perc_df(self) -> pd.DataFrame:
        """
        Converts DataFrame of weights into a DataFrame of percentages

        Returns:
            Weight percentage DataFrame
        """
        c = self.df.groupby(["tissue", "normal_tissue"])["Median"].sum().rename("count")
        perc = c / c.groupby(level=0).sum() * 100
        return perc.reset_index()

    def _load_tumor(self):
        print(f"Reading in {self.tumor_path}")
        if self.tumor_path.endswith(".csv"):
            df = pd.read_csv(self.tumor_path, index_col=0)
        elif self.tumor_path.endswith(".tsv"):
            df = pd.read_csv(self.tumor_path, sep="\t", index_col=0)
        else:
            try:
                df = pd.read_hdf(self.tumor_path)
            except Exception as e:
                print(e)
                raise RuntimeError(f"Failed to open DataFrame: {self.tumor_path}")
        return df

    def plot_match_scatter(self, out_dir: str = None):
        """
        Scatterplot of samples by tissue and their matched tissue model weight

        Args:
            out_dir: Optional output directory

        Returns:
            Plot axes object
        """
        df = self.df
        # Subset for matched-tissue samples
        df = df[df.normal_tissue == df.tissue].sort_values("tissue")

        f, ax = plt.subplots(figsize=(8, 4))
        sns.swarmplot(data=df, x="tissue", y="Median")
        plt.xticks(rotation=45)
        plt.xlabel("Tissue")
        plt.ylabel("GTEx Matched Tissue Weight")
        plt.title("TCGA Tumor Samples and Model Weight for GTEx Matched Tissue")
        if out_dir:
            plt.savefig(os.path.join(out_dir, "matched_weight_scatter.svg"))
        return ax

    def plot_perc_heatmap(self, out_dir: str = None):
        """
        Heatmap of weight percentages by

        Args:
            out_dir: Optional output directory

        Returns:
            Plot axes object
        """
        f, ax = plt.subplots(figsize=(8, 7))
        perc_heat = self.perc.pivot(
            index="normal_tissue", columns="tissue", values="count"
        )
        sns.heatmap(
            perc_heat.apply(lambda x: round(x, 2)),
            cmap="Blues",
            annot=True,
            linewidths=0.5,
        )
        plt.xlabel("Tumor Tissue")
        plt.ylabel("GTEx Tissue")
        plt.title(f"Average Weight (%) of Tumor to GTEx Tissue (n={self.num_samples})")
        if out_dir:
            plt.savefig(os.path.join(out_dir, "weight_perc_heatmap.svg"))
        return ax

    def plot_pca_nearby_tissues(self, background_path: str, tissues, tumor_tissue):
        df = pd.read_hdf(background_path)
        tumor = self.tumor
        tumor = tumor[tumor.tissue == tumor_tissue]
        tumor["tissue"] = f"{tumor_tissue}-Tumor"

        sub = df[df.tissue.isin(tissues)]
        pca_df = pd.concat([tumor, sub]).dropna(axis=1)
        embedding = pca.PCA(n_components=2).fit_transform(pca_df[self.genes])
        embedding = pd.DataFrame(embedding)
        embedding.columns = ["PCA1", "PCA2"]
        embedding["tissue"] = list(pca_df["tissue"])

        f, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(
            data=embedding, x="PCA1", y="PCA2", hue="tissue", style="tissue"
        )
        plt.title(f"PCA of {tumor_tissue} and Nearby GTEx Tissues")
        return ax

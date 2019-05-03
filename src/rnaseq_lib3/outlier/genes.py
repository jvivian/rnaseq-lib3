import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

sns.set_style("whitegrid")


class Genes:
    def __init__(
        self, sample_dir, background_path, sample_path, sample_name, genes_path
    ):
        self.sample_dir = sample_dir
        self.background_path = background_path
        self.sample_path = sample_path
        self.sample_name = sample_name
        self.w = self._weight_df()
        self.p = self._pval_df()
        self.traces = {}
        self.dfs = {}
        self.genes = [
            x.strip() for x in open(genes_path, "r").readlines() if not x.isspace()
        ]

    def _weight_df(self) -> pd.DataFrame:
        weights = []
        for subdir in os.listdir(self.sample_dir):
            sample_name, max_genes = subdir.split("_")
            weight_path = os.path.join(
                self.sample_dir, subdir, sample_name, "weights.tsv"
            )
            w = pd.read_csv(weight_path, sep="\t")
            w.columns = ["tissue", "Median", "std"]
            w["sample"] = sample_name
            w["max_genes"] = int(max_genes)
            w["num_training_genes"] = int(max_genes) - 85
            weights.append(w.drop("std", axis=1))
        weights = pd.concat(weights).reset_index(drop=True)
        return weights.sort_values("max_genes")

    def _pval_df(self) -> pd.DataFrame:
        pvals = []
        for subdir in os.listdir(self.sample_dir):
            sample_name, max_genes = subdir.split("_")
            pval_path = os.path.join(self.sample_dir, subdir, sample_name, "pvals.tsv")
            p = pd.read_csv(pval_path, sep="\t")
            p["sample"] = sample_name
            p["max_genes"] = int(max_genes)
            p["num_training_genes"] = int(max_genes) - 85
            pvals.append(p)
        pvals = pd.concat(pvals).reset_index(drop=True)
        return pvals.sort_values("max_genes")

    def _load_df(self, path):
        if path in self.dfs:
            return self.dfs[path]
        print(f"Reading in {path}")
        if path.endswith(".csv"):
            df = pd.read_csv(path, index_col=0)
        elif path.endswith(".tsv"):
            df = pd.read_csv(path, sep="\t", index_col=0)
        else:
            try:
                df = pd.read_hdf(path)
            except Exception as e:
                print(e)
                raise RuntimeError(f"Failed to open DataFrame: {path}")
        self.dfs[path] = df
        return df

    @staticmethod
    def _load_model(pkl_path):
        with open(pkl_path, "rb") as buff:
            data = pickle.load(buff)
        return data["model"], data["trace"]

    def _df(self, gene, tissue) -> pd.DataFrame:
        p = []
        for i, subdir in enumerate(os.listdir(self.sample_dir)):
            sample_name, max_genes = subdir.split("_")
            if subdir in self.traces:
                t = self.traces[subdir]
            else:
                if i == 0:
                    print("Loading traces to extract posterior distribution")
                model_path = os.path.join(
                    self.sample_dir, subdir, sample_name, "model.pkl"
                )
                m, t = self._load_model(model_path)
                self.traces[subdir] = t
            # Calculate PPC
            ppc = self._posterior_predictive_check(t, [gene])
            df = pd.DataFrame()
            df["ppc"] = ppc[gene]
            df["x"] = t[f"{gene}={tissue}"]
            df["max_genes"] = max_genes
            df["num_training_genes"] = int(max_genes) - 85
            p.append(df)
        p = pd.concat(p).reset_index(drop=True)
        return p.sort_values("max_genes")

    def _posterior_predictive_check(self, trace, genes):
        """
        Posterior predictive check for a list of genes trained in the model

        Args:
            trace: PyMC3 trace
            genes: List of genes of interest

        Returns:
            Dictionary of [genes, array of posterior sampling]
        """
        d = {}
        for gene in genes:
            d[gene] = self._gene_ppc(trace, gene)
        return d

    @staticmethod
    def _gene_ppc(trace, gene: str):
        """
        Calculate posterior predictive for a gene

        Args:
            trace: PyMC3 Trace
            gene: Gene of interest

        Returns:
            Random variates representing PPC of the gene
        """
        y_gene = [x for x in trace.varnames if x.startswith(f"{gene}=")]
        b = trace["a"]
        if "b" in trace.varnames:
            for i, y_name in enumerate(y_gene):
                b += trace["b"][:, i] * trace[y_name]

        # If no 'b' in trace.varnames then there was only one comparison group
        else:
            for i, y_name in enumerate(y_gene):
                b += 1 * trace[y_name]
        return np.random.laplace(loc=b, scale=trace["eps"])

    def _pearson_pvalue_matrix(self):
        df = self.p
        matrix = []
        df = df[df.Gene.isin(self.genes)]
        ntg_vector = df.num_training_genes.unique()
        for i in ntg_vector:
            row = []
            d1 = df[df.num_training_genes == i]
            d1.index = d1.Gene
            for j in ntg_vector:
                d2 = df[df.num_training_genes == j]
                d2.index = d2.Gene
                # Combine and index by gene
                c = pd.concat([d1.Pval, d2.Pval], axis=1)
                c.columns = [0, 1]
                r, _ = pearsonr(c[0], c[1])
                row.append(r)
            matrix.append(row)
        matrix = pd.DataFrame(matrix, index=ntg_vector, columns=ntg_vector)
        return matrix.apply(lambda x: round(x, 4))

    def plot_genes_by_weights(self):
        _, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=self.w, x="num_training_genes", y="Median", hue="tissue")
        plt.xlabel("Number of Additional Training Genes")
        plt.ylabel("Median Beta Coefficient")
        plt.title("Effect of Adding Training Genes on Beta Weights")
        return ax

    def plot_x(self, gene, tissue):
        bg = self._load_df(self.background_path)
        sg = self._load_df(self.sample_path)
        df = self._df(gene, tissue)

        _, ax = plt.subplots(figsize=(8, 5))
        plt.axvline(sg.loc[self.sample_name][gene], label="N-of-1", c="r")
        gr = df.groupby("num_training_genes")["x"]
        for label, arr in gr:
            sns.kdeplot(arr, label=label)
        sns.kdeplot(bg[bg.tissue == tissue][gene], shade=True, label="Observed")
        plt.xlabel("Transcripts per Million (log2(TPM +1))")
        plt.ylabel("Density")
        plt.title(f"Effect of n Genes on x Distribution - {gene}")
        return ax

    def plot_posterior(self, gene, tissue):
        sg = self._load_df(self.sample_path)
        bg = self._load_df(self.background_path)
        df = self._df(gene, tissue)

        _, ax = plt.subplots(figsize=(8, 5))
        plt.axvline(sg.loc[self.sample_name][gene], label="N-of-1", c="r")
        gr = df.groupby("num_training_genes")["ppc"]
        for label, arr in gr:
            sns.kdeplot(arr, label=label)
        sns.kdeplot(bg[bg.tissue == tissue][gene], shade=True, label="Observed")
        plt.xlabel("Transcripts per Million (log2(TPM +1))")
        plt.ylabel("Density")
        plt.title(f"Effect of n Genes on Posterior Distribution - {gene}")
        return ax

    def plot_pval_heatmap(self):
        matrix = self._pearson_pvalue_matrix()
        f, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(matrix, cmap="Blues", annot=True, linewidths=0.5)
        plt.xlabel("Number of Additional Training Genes")
        plt.ylabel("Number of Additional Training Genes")
        plt.title(
            "Pearson Correlation of Gene P-values (n=85) as Training Genes are Added"
        )
        return ax

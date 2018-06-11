from collections import defaultdict
from typing import List, Set

import pandas as pd

# TCGA Mapping
subtype_abbrev = {
    'LAML': 'Acute Myeloid Leukemia',
    'ACC': 'Adrenocortical carcinoma',
    'BLCA': 'Bladder Urothelial Carcinoma',
    'LGG': 'Brain Lower Grade Glioma',
    'BRCA': 'Breast invasive carcinoma',
    'CESC': 'Cervical squamous cell carcinoma and endocervical adenocarcinoma',
    'CHOL': 'Cholangiocarcinoma',
    'LCML': 'Chronic Myelogenous Leukemia',
    'COAD': 'Colon adenocarcinoma',
    'COADRED': 'Colorectal adenocarcinoma',
    'CNTL': 'Controls',
    'ESCA': 'Esophageal carcinoma',
    'FPPP': 'FFPE Pilot Phase II',
    'GBM': 'Glioblastoma multiforme',
    'HNSC': 'Head and Neck squamous cell carcinoma',
    'KICH': 'Kidney Chromophobe',
    'KIRC': 'Kidney renal clear cell carcinoma',
    'KIRP': 'Kidney renal papillary cell carcinoma',
    'LIHC': 'Liver hepatocellular carcinoma',
    'LUAD': 'Lung adenocarcinoma',
    'LUSC': 'Lung squamous cell carcinoma',
    'DLBC': 'Lymphoid Neoplasm Diffuse Large B-cell Lymphoma',
    'MESO': 'Mesothelioma',
    'MISC': 'Miscellaneous',
    'OV': 'Ovarian serous cystadenocarcinoma',
    'PAAD': 'Pancreatic adenocarcinoma',
    'PCPG': 'Pheochromocytoma and Paraganglioma',
    'PRAD': 'Prostate adenocarcinoma',
    'READ': 'Rectum adenocarcinoma',
    'SARC': 'Sarcoma',
    'SKCM': 'Skin Cutaneous Melanoma',
    'STAD': 'Stomach adenocarcinoma',
    'TGCT': 'Testicular Germ Cell Tumors',
    'THYM': 'Thymoma',
    'THCA': 'Thyroid carcinoma',
    'UCS': 'Uterine Carcinosarcoma',
    'UCEC': 'Uterine Corpus Endometrial Carcinoma',
    'UVM': 'Uveal Melanoma',
}


# TCGA SNV Functions
def mutations_for_gene(driver_mutations_path: str, gene: str) -> List[str]:
    """Returns set of mutations for a TCGA cancer driver gene"""
    mut = pd.read_csv(driver_mutations_path, sep='\t')
    mut_set = mut[mut.Gene == gene].Mutation.unique()
    return sorted(mut_set)


def subtypes_for_gene(driver_consensus_path: str, gene: str) -> List[str]:
    """Returns TCGA cancer subtypes for a given gene"""
    con = pd.read_csv(driver_consensus_path, sep='\t')
    cancer_set = con[con.Gene == gene].Cancer.unique()
    submap = subtype_abbrev
    cancer_mapping = [submap[x] if x in submap else x for x in cancer_set]
    cancer_mapping = ['_'.join(y.capitalize() for y in x.split()) for x in cancer_mapping]
    return cancer_mapping


def subtype_filter(metadata: pd.DataFrame, samples: List[str], subtypes: List[str]) -> Set[str]:
    """Filter samples by set of valid subtypes"""
    sub = metadata[metadata.type.isin(subtypes)]
    return set(samples).intersection(set(sub.id))


def mutation_sample_map(snv: pd.DataFrame, gene: str, mutations: List[str]) -> pd.DataFrame:
    """Identify samples with a given set of mutations in a particular gene"""
    # Subset by variant type and mutation
    sub = snv[(snv.SYMBOL == gene) & (snv.Variant_Type == 'SNP')]

    # Collect samples for all mutants
    s = defaultdict(set)
    for mut in mutations:
        s[mut].update([x[:15] for x in sub[sub.HGVSp_Short == mut].Tumor_Sample_Barcode])

    # Convert to DataFrame
    df = pd.DataFrame(list({x: k for k, v in s.items() for x in v}.items()), columns=['Sample', 'Mutation'])
    df = df.set_index('Sample')
    df.index.name = None
    return df


def pathway_from_gene(driver_pathway_path: str, gene: str) -> str:
    """Returns TCGA cancer driver pathway for a given gene"""
    path = pd.read_csv(driver_pathway_path, sep='\t')
    pathway = path[path.Gene == gene].Pathway.unique()
    if len(pathway) != 1:
        print(f'More than 1 pathway found: {pathway}')
        return pathway
    else:
        return pathway[0]

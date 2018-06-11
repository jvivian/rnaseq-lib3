from typing import List


def get_protein_coding_genes(gtf: str) -> List[str]:
    """Collect protein-coding genes from GTF file"""
    genes = []
    with open(gtf, 'r') as f:
        for line in f:
            line = line.split('"')
            try:
                if line[3] == 'protein_coding':
                    genes.append(line[1])
            except IndexError:
                pass
    return genes

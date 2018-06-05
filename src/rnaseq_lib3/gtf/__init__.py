def get_protein_coding_genes(gtf):
    """
    Collect protein-coding genes from GTF file

    :param str gtf: Path to GTF file to parse
    :return: Protein coding genes
    :rtype: list(str)
    """
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

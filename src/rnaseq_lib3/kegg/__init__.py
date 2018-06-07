import io
import os

from Bio.KEGG.KGML import KGML_parser

from src.rnaseq_lib3.utils import rget


def get_genes_from_pathway(pathway: str) -> set:
    kgml = _get(pathway, form='kgml').text

    # Wrap text in a file handle for KGML parser
    f = io.BytesIO(kgml.encode('utf-8'))
    k = KGML_parser.read(f)

    genes = set()
    for gene in k.genes:
        for x in gene.name.split():
            g = get_gene_names_from_label(x)
            genes = genes.union(g)

    return genes


def get_gene_names_from_label(label: str) -> set:
    genes = set()
    r = _get(label)
    for line in r.text.split('\n'):
        if line.startswith('NAME'):
            line = line.split()[1:]
            genes.add(line[0].rstrip(','))
    return genes


def _get(query: str, form=None):
    return _kegg_search(operation='get', database='', query=query, form=form)


def _kegg_search(operation: str, database: str, query=None, form=None):
    # Set arguments to empty strings if None
    query = '' if query is None else query
    form = '' if form is None else form

    # Define base URL
    url = 'http://rest.kegg.jp'

    # Make get request
    request = os.path.join(url, operation, database, query, form)
    return rget(request)

import io
import os

from Bio.KEGG.KGML import KGML_parser

from src.rnaseq_lib3.utils import rget


def get_genes_from_pathway(pathway: str) -> set:
    # Find pathway name
    kegg_path = _find(database='pathway', query=pathway).text
    if kegg_path == '\n':
        raise RuntimeError(f'Pathway {pathway} not found')

    # Extract HSA version of pathway
    kegg_path, description = kegg_path.split('\t')
    kegg_path = 'hsa' + ''.join(x for x in kegg_path if x.isnumeric())

    kgml = _get(kegg_path, form='kgml').text

    # Wrap text in a file handle for KGML parser
    f = io.StringIO(kgml)
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


def _find(database: str, query: str):
    return _kegg_search(operation='find', database=database, query=query)


def _get(query: str, database=None, form=None):
    return _kegg_search(operation='get', database=database, query=query, form=form)


def _kegg_search(operation: str, database=None, query=None, form=None):
    # Set arguments to empty strings if None
    query = '' if query is None else query
    form = '' if form is None else form
    database = '' if database is None else database

    # Define base URL
    url = 'http://rest.kegg.jp'

    # Make get request
    request = os.path.join(url, operation, database, query, form)
    return rget(request)

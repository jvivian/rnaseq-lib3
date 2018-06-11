import io
import os

from Bio.KEGG.KGML import KGML_parser
from requests.models import Response

from rnaseq_lib3.utils import rget


def pathway_genes(pathway: str) -> set:
    """Returns genes for a given pathway in KEGG"""
    kgml = _get(pathway, form='kgml').text

    # Wrap text in a file handle for KGML parser
    f = io.StringIO(kgml)
    k = KGML_parser.read(f)

    genes = set()
    for gene in k.genes:
        for x in gene.name.split():
            g = kegg_label_gene_names(x)
            genes = genes.union(g)

    return genes


def kegg_label_gene_names(label: str) -> set:
    """Extracts gene name from KEGG label"""
    genes = set()
    r = _get(label)
    for line in r.text.split('\n'):
        if line.startswith('NAME'):
            line = line.split()[1:]
            genes.add(line[0].rstrip(','))
    return genes


def _find(database: str, query: str) -> Response:
    """Kegg 'find' wrapper"""
    return _kegg_query(operation='find', database=database, query=query)


def _get(query: str, database: str = None, form: str = None) -> Response:
    """KEGG 'get' wrapper"""
    return _kegg_query(operation='get', database=database, query=query, form=form)


def _kegg_query(operation: str, database: str = None, query: str = None, form: str = None) -> Response:
    """Runs Kegg API query"""
    # Set arguments to empty strings if None
    query = '' if query is None else query
    form = '' if form is None else form
    database = '' if database is None else database

    # Define base URL
    url = 'http://rest.kegg.jp'

    # Make get request
    request = os.path.join(url, operation, database, query, form)
    return rget(request)

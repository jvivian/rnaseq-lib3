import os

from synapseclient import Synapse, File

expression = 'syn11311347'
metadata = 'syn11311931'
objects = 'syn11515014'
pathway = 'syn12523802'


def upload_file(file_path: str, login: str, parent: str, description: str = None) -> None:
    """Uploads file to Synapse. Password must be stored in environment variable SYNAPSE_PASS"""
    description = '' if None else description
    f = File(file_path, description=description, parent=parent)

    syn = _syn_login(login)
    syn.store(f)


def download_file(syn_id: str, login: str, download_location: str = '.') -> None:
    """Synapse ID of file to download. Password must be stored in environment variable SYNAPSE_PASS"""
    syn = _syn_login(login)
    syn.get(syn_id, downloadLocation=download_location)


def _syn_login(login: str) -> Synapse:
    """Login to synapse. Password must be stored in environment variable SYNAPSE_PASS"""
    assert 'SYNAPSE_PASS' in os.environ, 'SYNAPSE_PASS must be set as an environment variable'
    syn = Synapse()
    syn.login(login, os.environ['SYNAPSE_PASS'])
    return syn


# TODO: Finish
def mirror_synapse_dir(syn_id: str, login: str):
    """Mirror a synapse directory's contents. Password must be stored in environment variable SYNAPSE_PASS"""
    pass

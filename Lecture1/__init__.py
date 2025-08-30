import os
import html
from urllib.parse import urljoin
import ROOT
from ipywidgets import interact
from IPython.display import display, HTML

def vscode(path, target="_blank"):
    """
    Generates an HTML link to open a directory in VS Code within JupyterHub.

    Parameters:
    path (str): The relative or absolute path to the file or directory.
    target (str): The target attribute for the <a> tag, default is "_blank".
    """
    abspath = os.path.abspath(path)
    route = os.getenv('JUPYTERHUB_SERVICE_PREFIX') or '/'
    route = urljoin(route, 'vscode/')
    route += f'?folder={abspath}'
    link_text = f'Click to open in VS Code: {abspath}'
    display(HTML(f'<a href={html.escape(route)} target="{html.escape(target)}" title={route}>{link_text}</a>'))

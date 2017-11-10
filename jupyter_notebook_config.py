### If you want to auto-save .html and .py versions of your notebook:
# Reference: https://svds.com/jupyter-notebook-best-practices-for-data-science/
c = get_config()
import os
from subprocess import check_call

def post_save(model, os_path, contents_manager):
    """post-save hook for converting notebooks to .py scripts"""
    if model['type'] != 'notebook':
        return # only do this for notebooks
    d, fname = os.path.split(os_path)
    check_call(['jupyter', 'nbconvert', '--to', 'script', fname], cwd=d)
    check_call(['jupyter', 'nbconvert', '--to', 'html', fname], cwd=d)
    os.chown(d+"/"+fname,1000,1000)
    basename,_ = os.path.splitext(fname)
    os.chown(d+"/"+basename+".py",1000,1000)
    os.chown(d+"/"+basename+".html",1000,1000)

c.FileContentsManager.post_save_hook = post_save

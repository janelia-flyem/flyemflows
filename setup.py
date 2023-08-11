"""
flyemflows setup.py
"""
from setuptools import find_packages, setup
import versioneer

# For now, requirements are only specified in the conda recipe, not here.
#
# TODO: Specify them here (or requirements.txt),
#       and have the conda recipe import them via:
#
#    run:
#      - python
#       {% for dep in data['install_requires'] %}
#      - {{ dep.lower() }}
#      {% endfor %}
#
setup(
    name='flyemflows',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Dask-based reconstruction tools working large image volumes, especially for DVID',
    url='https://github.com/janelia-flyem/flyemflows',
    packages=find_packages(exclude=('tests')),
    package_data={'flyemflows': ['scripts/*']},
    entry_points={
        'console_scripts': [
            'launchflow = flyemflows.bin.launchflow:main',
            'init_dvid_bucket = flyemflows.bin.init_dvid_bucket:main',
            'ingest_label_indexes = flyemflows.bin.ingest_label_indexes:main',
            'erase_from_labelindexes = flyemflows.bin.erase_from_labelindexes:main',
            'mesh_update_daemon = flyemflows.bin.mesh_update_daemon:main',
            'roistats_table = flyemflows.workflow.util.roistats_table:main',
        ]
    }
)

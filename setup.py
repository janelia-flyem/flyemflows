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
setup( name='flyemflows',
       version=versioneer.get_version(),
       cmdclass=versioneer.get_cmdclass(),
       description='Dask-based reconstruction tools working large image volumes, especially for DVID',
       url='https://github.com/janelia-flyem/flyemflows',
       packages=find_packages(exclude=('tests')),
       package_data={'flyemflows': ['scripts/*']},
       entry_points={
          'console_scripts': [
              'launchflow = flyemflows.bin.launchflow:main'
          ]
       }
     )

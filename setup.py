from distutils.core import setup
from setuptools import find_packages

VERSION = '0.1'

setup(
    name='local_llm',
    description='Deploy your own local LLM',
    version=VERSION,
    include_package_data=True,
    packages=find_packages(),
    license='GNU3',
    platforms='any',
    entry_points={
        'console_scripts': [
            'process_documents=llm.process_documents:index_documents',
        ],
    },
)

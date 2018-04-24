from pip.req import parse_requirements
from setuptools import setup, find_packages

install_reqs = parse_requirements('requirements.txt')

setup(
    name='RFFT',
    version='0.0.1',
    author='Amogh Mannekote',
    author_email='msamogh@gmail.com',
    description='Right For The First Time',
    license='MIT',
    url='http://github.com/msamogh/RFFT',
    packages=find_packages(),
    install_reqs=install_reqs
)

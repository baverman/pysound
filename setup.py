from setuptools import setup, find_packages
from setuptools.extension import Extension

ext_map = {
    'filters': ['filters.c'],
}

extensions = [Extension(k, v) for k, v in ext_map.items()]

setup(
    name='pysound',
    version='0.1',
    url='https://github.com/baverman/pysound',
    author='Anton Bobrov',
    author_email='baverman@gmail.com',
    license='MIT',
    description='Sound syntesis',
    # long_description=open('README.rst').read(),
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy'],
    ext_modules=extensions,
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ]
)

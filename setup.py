from setuptools import setup, find_packages

setup(
    name='pipeline_optimizer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
    ],
    author='Stanislav Kharchenko',
    author_email='stanislakh@gmail.com',
    description='A package for creating sequential transformers for machine learning pipelines',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sk8997/pipeline-optimizer',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
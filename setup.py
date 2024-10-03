from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='contextual_retrieval',
    version='0.1.1',
    description='An open-source Python RAG library for Contextual Retrieval',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Hamada Fadil Mahdi',
    author_email='hamadamahdi2014@gmail.com',
    url='https://github.com/HamadaFMahdi/contextual-retrieval',
    packages=find_packages(exclude=["tests", "examples"]),
    install_requires=[
        'aiohttp>=3.7.4',
        'faiss-cpu>=1.7.2',
        'nltk>=3.6.2',
        'numpy>=1.21.0',
        'openai>=0.27.0',
        'rank_bm25>=0.2.2',
        'requests>=2.25.1',
        'sentence-transformers>=2.2.0',
        'torch>=1.9.0',
        'tqdm>=4.62.0',
        'transformers>=4.10.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
)

# python setup.py sdist bdist_wheel && twine upload --config-file ./.pypirc dist/*
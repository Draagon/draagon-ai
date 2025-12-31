"""Setup file for editable install."""

from setuptools import setup, find_packages

setup(
    name="implicit_knowledge_graphs",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)

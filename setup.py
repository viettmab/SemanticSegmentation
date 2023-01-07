from setuptools import setup

setup(
    name="SemanticSegmentation",
    py_modules=["SemanticSegmentation"],
    install_requires=[ "torch", "tqdm"],
)
import setuptools

setuptools.setup(
    name="line_planning",
    version="0.1.0",
    author="David Fiedler",
    author_email="david.fido.fiedler@gmail.com",
    description="Line Planning",
    packages=setuptools.find_packages(),
    install_requires=[
        "gurobipy",
    ],
)
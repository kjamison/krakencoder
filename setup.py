import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = [r for r in f.read().splitlines() if not r.startswith("#")]

with open('krakencoder/_version.py') as f:
    version=f.readline().split("=")[-1].split()[-1].replace("'","")

setuptools.setup(
    name="krakencoder", # Replace with your own username
    version=version,
    author="Keith Jamison",
    author_email="keith.jamison@gmail.com",
    description="Implementation of a joint connectome mapping tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kjamison/krakencoder",
    packages=setuptools.find_packages(),
    install_requires = required,
    include_package_data = True,
    package_data={
        "krakencoder": ["resources/*"],
        "tests": ["resources/*"],
    },
    classifiers= [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
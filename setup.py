from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as f:
    required = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="auto5hmcML",
    version="0.1.0",
    author="SHY",
    author_email="serine94209792@gmail.com",
    url="https://guthub.com/Serine94209792/auto5hmcML",
    packages=find_packages(),
    description="autoML for 5hmcseq",
    long_description=open("README.md", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    license="MIT",
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
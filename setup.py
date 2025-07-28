from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="threadward",
    version="0.1.0",
    author="threadward",
    description="Parallel Processing for Generalizable AI Experimentation in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mamarcus64/threadward",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "psutil>=5.9.0",
        "GPUtil>=1.4.0",
    ],
    entry_points={
        "console_scripts": [
            "threadward=threadward.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "threadward": ["templates/*"],
    },
)
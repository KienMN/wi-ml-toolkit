import os
from setuptools import setup, find_packages

path = os.path.abspath(os.path.dirname(__file__))

readme = open(path + "/docs/README.md")

setup(
  name="wi-ml-toolkit",
  version="1.0.0",
  description="A toolkit for classification and regression problems",
  url="https://github.com/KienMN/wi-ml-tookit",
  author="WI ML Team",
  author_email="kienmn97@gmail.com",
  license="MIT",
  packages=find_packages(exclude=["docs","tests", ".gitignore"]),
  install_requires=["numpy", "pandas", "scikit-learn", "keras", "scipy"],
  dependency_links=[""],
  include_package_data=True
)
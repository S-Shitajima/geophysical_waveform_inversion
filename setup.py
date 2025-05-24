from setuptools import find_packages, setup


setup(
    name="geophysical_waveform_inversion",             # パッケージ名（任意）
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},         # パッケージのルートディレクトリを'src'に指定
    author="ss", 
    description="A sample packages",
)
from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        rquirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='zomato_project',
    version='0.0.1',
    description='Time taken per delivery prediction for Zomato',
    author='Abhishek',
    author_email='abhishekdutta.9579@gmail.com',
    url='https://github.com/sujitmandal/py-data-api',
    license='MIT',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")


)
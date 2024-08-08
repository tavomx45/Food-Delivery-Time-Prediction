from setuptools import setup, find_packages
from typing import List

# function to load the requirements from the file
 
HYPEN_E_DOT = '-e .'

def get_requirements() -> List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open("requirements.txt") as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(name='Food Delivery Time Prediction',
      version='0.0.1',
      description='Use of Machine Learning to Predict the food delivery time',
      author='Gustavo Paredes',
      author_email='bryangustavoparedes@gmail.com',
      packages=find_packages(),
      install_requires= get_requirements()
     )
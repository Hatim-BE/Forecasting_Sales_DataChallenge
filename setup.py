from setuptools import find_packages, setup

def get_requirements(path_file:str)->list:
    '''
    This function will return the 
    requirements in a form of a list
    written on the specified path_file
    '''

    requirements = []
    with open(path_file) as f:
        for line in f:
            line = line.strip() #delete any leading/trailing whitespaces / newlines
            if not line or line == "-e .": continue
            requirements.append(line)

    return requirements


setup(
    name='MLOps_project',
    version='0.1.0',
    description='A first ever MLOps project',
    author='Hatim Belfarhounia',
    author_email='nothatim@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
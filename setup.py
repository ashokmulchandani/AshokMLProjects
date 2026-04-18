from setuptools import find_packages, setup

def get_requirements(file_path):
    '''
    This function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirement]
        
    return requirements

HYPEN_E_DOT='-e.'

if HYPEN_E_DOT in requirement:
        requirement.remove(HYPEN_E_DOT)
setup(
    name='mlproject',
    version='0.0.1',
    packages=find_packages(),
    auther='Ashok Mulchandani',
    auther_email='ashokmulchandani@gmail.com'
    install_requires=get_requirements('requirement.txt')       
)

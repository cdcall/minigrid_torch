from setuptools import setup, find_packages

setup(
    name='rl_sandbox',
    version='0.0.1',
    keywords='agent, rl, gym-minigrid, torch_ac, train, visualize, evaluate',
    url='https://github.com/drucej/minigrid_torch',
    description='RL starter files to train, visualize and evaluate an agent **without writing any line of code**',
    packages=find_packages(exclude=(".idea", ".ipynb_checkpoints", "storage")),
    install_requires=[
        'torch>=1.7.0'
        'numpy==1.18.5',
        'gym>=0.9.6',
        'matplotlib',
        'six>=1.12.0',
        'array2gif',
        'jupyter',
        'tensorboardX>=1.6',
        'tensorboard>=2.4',
        'tensorflow==2.3.1'
    ]
)

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='Charmlang',
    version='0.8a1',
    author='Weilong Cui and UCSB ArchLab',
    author_email='cuiwl@cs.ucsb.edu',
    description='A language for closed-form high-level architecture modeling',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/UCSBarchlab/Charm',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
)

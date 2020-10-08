import setuptools

with open('README.rst', 'r') as f:
    readme = f.read()

setuptools.setup(
	name='resreg',
	version='0.2',
	author='Japheth Gado',
	author_email='japhethgado@gmail.com',
	description='Resampling strategies for regression',
	long_description_content_type='text/x-rst',
	long_description=readme,
	url='https://github.com/jafetgado/resreg',
	keywords='resampling regression machine-learning preprocessing',
	packages=setuptools.find_packages(),
	include_package_data=True,
	license='GNU GPLv3',
	classifiers=[
		'Programming Language :: Python :: 3',
		'Operating System :: OS Independent',
		'Topic :: Scientific/Engineering'
				],
	install_requires=['numpy', 'pandas', 'scipy', 'scikit-learn'],
	python_requires='>=3'
		)

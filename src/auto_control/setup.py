from setuptools import find_packages, setup

package_name = 'auto_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shuiyihang',
    maintainer_email='shuiyihang0981@163.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'test_get_state=auto_control.get_state:main',
            'track_car_train=auto_control.track_car_train:main',
            'track_car_test=auto_control.track_car_test:main'
        ],
    },
)

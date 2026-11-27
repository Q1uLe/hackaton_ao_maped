from glob import glob

from setuptools import find_packages, setup

package_name = 'aruko'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='q1ule',
    maintainer_email='sar1p013pehat@gmail.com',
    description='T5 navigation and ArUco detection',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'detector = aruko.detector:main',
            'course_odometry = aruko.course_odometry:main',
            'scan_filter = aruko.scan_filter:main',
            'planning_map = aruko.planning_map:main',
        ],
    },
)

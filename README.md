# SLAM_project
## Main function:
This code loads sensors' data, combined analysis readings from encoder, 2D lidar and FOG sensor, and generate map of surrounding environment of robot.
## Usage
1. create "sensing_data" file </br>
2. change the parameter in map_init.py</br>
3. add sensors' data file to directory </br>
4. run main.py
## Sensing Data Usage:
Encoder: 2 * n, left wheel, right wheel</br>
FOG: 3 * n, delta angle in x, y, z</br>
2D-Lidar: 286 * n, obstcle location in different directions</br>
</br>
## Output:
![image](https://user-images.githubusercontent.com/42241063/156458967-8ca6c6e6-b230-46c9-a8d1-4074050fef7e.png) </br>
SLAM mapping

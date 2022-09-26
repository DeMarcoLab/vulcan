
@patrickcleeve2

START_FROM:
- turn on microscope ops

DONE:

show microscope settings
connected, tilt, gas, etc
move to milling angle... etc

load profile as npy
set settings: width, height , depth
run optional initial milling

options to transpose, invert? Dont think we need transpose?
invert profile
convert to bmp

set top left corner position
move to corner
move to centre


TODO:

set milling position
set milling pattern
draw milling pattern in napari
take ion image
change to milling current on setup 
run milling

offset movement
corner edge detection

profile decomposition
multiple milling stages
stitching


calibration
- run calibration pattern
- move to edge, milling multiple rectangles at different depths, see how deep they go, fit the actual sputtering rate / depth
- actually set to an application file??

STRETCH: 
    - 3d profiles

docs
- setting patterns
- understanding bitmap millng
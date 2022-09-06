import math
import random
import sys
import time
from importlib import invalidate_caches
from io import BytesIO
from typing import Tuple

import cv2
import numpy as np
import requests
from PIL import Image

from google_api_key import api_key


class RoadBEVGenerator():
    '''
    A program for generating road BEV representations (np.array) from Google
    Maps API rasterizations.

    How to use:

        # Initialize
        view_size_m = 80
        view_size_px = 512
        road_bev_gen = RoadBEVGenerator(view_size_m, view_size_px)

        # Specify center point (Google maps coordinate)
        p1 = (1.2973085447791457, 103.7903854091712)

        # Generate road BEV np.array
        road_bev = road_bev_gen.generate(p1)
    '''
    def __init__(self, view_size_meter:float, view_size_px:int, zoom:int=19):
        '''
        Args:
            view_size_meter: Physical road view dimension (i.e. 80x80 meters).
            view_size_px: Road view dimension in image (i.e. 256x256 pixels).
            zoom:  Integer which defines the resolution of the current view.
                    0: The lowest zoom level for the entire world.
        '''
        self.zoom = zoom

        self.view_size_meter = view_size_meter
        self.view_size_px = view_size_px

        self.view_size = (self.view_size_px, self.view_size_px)
        self.gmap_size = (1280, 1280)  # Maximum size for standard plan

        # Center point coordinates
        self.gmap_center_w = self.gmap_size[0] // 2
        self.gmap_center_h = self.gmap_size[1] // 2
        self.view_center_w = self.view_size[0] // 2
        self.view_center_h = self.view_size[1] // 2

        self.road_thres_low = 128
        self.road_thres_high = 255

        self.gmap_m_per_px = self.gmap_m_per_px_ratio(self.zoom)

    def generate(self, pnt_1:Tuple, connected_only:bool=False) -> np.array:
        '''
        Generates a road BEV raster representation centered on the given point.

        Args:
            p1: Map point (latitude, longitude) coordinates.
            connected_only: Generate only the road region connected to the
                            specified point (i.e. ignore disconnected
                            opposite lanes).
        
        Returns:
            Boolean matrix representing the road BEV.
        '''
        # Generate a path between two points on the road to define a heading
        #try:
        pnt_1 = self.snap_to_road([pnt_1])
        if len(pnt_1) == 0:
            return None
        pnt_1 = pnt_1[0]

        th = random.uniform(0,360)
        pnt_2 = [pnt_1[0]+0.00001*math.sin(th), pnt_1[1]+0.00001*math.cos(th)]
        pnt_2 = self.snap_to_road([pnt_2])
        if len(pnt_2) == 0:
            return None
        pnt_2 = pnt_2[0]

        snapped_path = np.stack((pnt_1, pnt_2))

        # Generate raw raster map
        gmap = self.get_gmap(snapped_path[0])

        # Rotate map to 'upwards' heading using pixel-based vector angle
        dx_px, dy_px = self.loc2pixel(
            snapped_path[1] - snapped_path[0], self.zoom)
        angle = self.cal_directional_angle(dx_px, dy_px)
        gmap = self.rotate_map(gmap, angle)

        # Clip map to specified spatial dimension
        gmap_cut_size = int(np.round(
            self.view_size_meter / self.gmap_m_per_px))
        gmap = self.center_clip_map(gmap, gmap_cut_size, gmap_cut_size)

        # Generate filled contour map
        gmap = cv2.cvtColor(gmap, cv2.COLOR_RGB2GRAY)
        ret, gmap = cv2.threshold(
            gmap, self.road_thres_low, self.road_thres_high, cv2.THRESH_BINARY)
        gmap = self.reduce_edge_noise(gmap)
        
        # Removes all road regions not connected to the specified point
        if connected_only:
            contours, _ = cv2.findContours(
                gmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            gmap_new = np.zeros_like(gmap, dtype=np.uint8)
            isPointOnRoad = False
            for contour in contours:
                is_inside = cv2.pointPolygonTest(
                    contour, (gmap_cut_size//2, gmap_cut_size//2), False)
                if is_inside == 1:
                    # Draws filled contour
                    cv2.drawContours(gmap_new, [contour], -1, 255, -1)
                    isPointOnRoad = True

            if isPointOnRoad==False:
                print('Cannot find road')
                return None
            
            gmap = gmap_new

        # Resize the x [m] map to specified dimension pixel dimension
        gmap = cv2.resize(gmap,
                          (self.view_size_px, self.view_size_px),
                          interpolation=cv2.INTER_NEAREST)

        return gmap

    @staticmethod
    def reduce_edge_noise(binary_map):
        kernel = np.ones((3,3), dtype=np.uint8)
        binary_map = cv2.dilate(binary_map, kernel, iterations=1)
        binary_map = cv2.erode(binary_map, kernel, iterations=1)
        binary_map = cv2.erode(binary_map, kernel, iterations=1)
        binary_map = cv2.dilate(binary_map, kernel, iterations=1)
        return binary_map

    def snap_to_road(self, path:Tuple) -> np.array:
        '''
        Args:
            path: [pnt1, pnt2]
        
        Returns:
            Row vector matrix of coordinate tuples (x,y) representing points on
            the road.
                Ex: snapped_path[0] --> [x, y] for point 0

        '''
        path_str = ''
        for p in path:
            path_str += self.loc2str(p)+'|'
        path_str = path_str[:-1]

        url = 'https://roads.googleapis.com/v1/nearestRoads?points=' \
              + path_str +'&key=' + api_key
        payload={}
        headers = {}
        res = requests.request("GET", url, headers=headers, data=payload)

        snapped_path = []
        if 'snappedPoints' not in res.json().keys():
            print('ERROR: Could not snap point to road')
            return []
        for loc in res.json()['snappedPoints']:
            p = [loc['location']['latitude'], loc['location']['longitude']]
            if len(snapped_path)>0:
                if p[0] != snapped_path[-1][0] and p[1] != snapped_path[-1][1]:
                    snapped_path.append(p)
            else:
                snapped_path.append(p)
        snapped_path = np.array(snapped_path)
        return snapped_path

    def get_gmap(self, pnt):
        '''
        Creates an RGB image with Google Maps road semantics centered on a
        given point.

        NOTE: Modifies size to account for scale=2 (double resolution).

        Ref: https://developers.google.com/maps/documentation/maps-static/start

        Args:
            pnt: Center point as np.array with (latitude,longitude) values.
        
        Returns:
            RGB image as np.array (H,W,3).
        '''
        pnt_str = self.loc2str(pnt)
        size_str = '{}x{}'.format(self.gmap_size[0]//2, self.gmap_size[1]//2)
        url = 'https://maps.googleapis.com/maps/api/staticmap' \
              + '?center=' + pnt_str \
              + '&zoom=' + str(self.zoom) \
              + '&size=' + size_str \
              + '&scale=' + str(2) \
              + '&key=' + api_key \
              + '&style=feature:all|visibility:off' \
              + '&style=feature:road|element:geometry.fill|color:0xffffff|' \
              + 'visibility:on' \

        res = requests.get(url)

        try:
            gmap = Image.open(BytesIO(res.content)).convert('RGB')
        except Exception as e:
            print(e)
            return None

        gmap = np.array(gmap, dtype=np.uint8)
        return gmap

    def rotate_map(self, map:np.array, angle:float) -> np.array:
        '''
        Args:
            map: (H, W, 1|3)
            angle: Rotation angle in degrees.
        '''
        trans = cv2.getRotationMatrix2D(
            (self.gmap_center_w, self.gmap_center_h), angle, 1.0)
        map = cv2.warpAffine(
            map, trans, (self.gmap_size[0],self.gmap_size[1]))
        return map

    def center_clip_map(self, map:np.array, width:int, height:int) -> np.array:
        '''
        Args:
            map: (H, W, 1|3)
            width: Clip box width.
            height: Clip box height.
        '''
        i_0 = self.gmap_center_h - height // 2
        i_1 = self.gmap_center_h + height // 2
        j_0 = self.gmap_center_w - width // 2
        j_1 = self.gmap_center_w + width // 2
        map = map[i_0:i_1, j_0:j_1]
        return map
    
    def get_m_per_px(self):
        return self.gmap_m_per_px

    def get_gmap_size_m(self):
        return self.gmap_m_per_px*1280

    @staticmethod
    def loc2str(p):
        str = '{:.6f},{:.6f}'.format(p[0], p[1])
        return str

    @staticmethod
    def loc2pixel(p, zoom, scale=2):
        return [int(x*(2**zoom * scale)) for x in p]
    
    @staticmethod
    def gmap_m_per_px_ratio(zoom, scale=2):
        earth_circumference = 4.0075017e7 #  [m]
        zoom_z_map_size = 256  # [px]
        m_per_px = earth_circumference / (zoom_z_map_size*math.pow(2, zoom))
        # Account for doubled resolution
        m_per_px /= scale
        return m_per_px

    @staticmethod
    def cal_directional_angle(dx, dy, degrees=True):
        angle = math.atan2(dy, dx)
        if degrees:
            angle = math.degrees(angle)
        return angle


if __name__ == '__main__':

    view_size_m = 80
    view_size_px = 512
    road_bev_gen = RoadBEVGenerator(view_size_m, view_size_px)

    print('')
    print(f'gmap size: 1280 px == {road_bev_gen.get_gmap_size_m():.2f} m')
    print(f'm_per_px: {road_bev_gen.get_m_per_px():.4f} m/px')
    print('')

    p1 = (1.2973085447791457, 103.7903854091712)

    road_bev = road_bev_gen.generate(p1)

    cv2.imwrite('road_bev.png', road_bev)

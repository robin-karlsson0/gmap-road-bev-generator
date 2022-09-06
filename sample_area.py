import argparse
import os
import random
from typing import Tuple

import cv2
import numpy as np

from road_bev_generator import RoadBEVGenerator


def sample_area(
        view_size_m:float, view_size_px:int, tl_coord:Tuple, br_coord:Tuple,
        num_samples:int, area_name:str, subdir_size:int=1000):
    '''
    Generate N views by randomly sampling an area defined by top-left and 
    bottom-right coordinates (obtained by right-clicking on Google Maps etc.)

    Args:
        view_size_meter: Physical road view dimension (i.e. 80x80 meters).
        view_size_px: Road view dimension in image (i.e. 256x256 pixels).
        tl_coord: Coordinate pair of top-left region bounding box.
                  Ex: (1.30671226, 103.783898)
        br_coord: Coordinate pair of bottom_right region bounding box.
                  Ex: (1.29045516,103.804646)
        num_sampls: Number of BEV samples to randomly generate within region.
    '''
    road_bev_gen = RoadBEVGenerator(view_size_m, view_size_px)

    sample_idx = 0
    subdir_idx = 0
    for abs_sample_idx in range(num_samples):

        # Try generating a BEV until sampled coordinates work
        # Note order due to (latitude, longitude) coordinate system
        while True:
            
            coord_0 = random.uniform(tl_coord[0], br_coord[0])
            coord_1 = random.uniform(br_coord[1], tl_coord[1])
            coord = (coord_0, coord_1)
            print(f'idx {abs_sample_idx} | ({coord_0:.3f}, {coord_1:.3f})')

            road_bev = road_bev_gen.generate(coord)
            if road_bev is not None:
                break

        if sample_idx > subdir_size:
            sample_idx = 0
            subdir_idx += 1
        
        filename = f'{sample_idx}.png'
        sample_dir = f'./{area_name}/subdir{subdir_idx:03d}/'

        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)

        sample_path = os.path.join(sample_dir, filename)
        cv2.imwrite(sample_path, road_bev)

        sample_idx += 1


def parse_query_str(query):
    tl_coord, br_coord, num_samples, area_name = query.split('_')

    tl_coord = tl_coord.split(',')
    tl_coord = tuple([float(coord) for coord in tl_coord])

    br_coord = br_coord.split(',')
    br_coord = tuple([float(coord) for coord in br_coord])

    num_samples = int(num_samples)

    return tl_coord, br_coord, num_samples, area_name


if __name__ == '__main__':
    '''
    Query string format
        'top-left coords'_'bottom_right coords'_'sample number'_'area name'
        1.30671226,103.783898_1.29045516,103.804646_10000
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('view_size_m',
                        type=float,
                        help='Spatial extent of generated views [m]')
    parser.add_argument('view_size_px',
                        type=int,
                        help='Image dimension of generated views [px]')
    parser.add_argument(
        '--query', type=str, nargs='+', help='Sequence of query strings')
    args = parser.parse_args()

    view_size_m = args.view_size_m
    view_size_px = args.view_size_px

    for query in args.query:
        
        tl_coord, br_coord, num_samples, area_name = parse_query_str(query)

        sample_area(
            view_size_m, view_size_px, tl_coord, br_coord, num_samples,
            area_name)



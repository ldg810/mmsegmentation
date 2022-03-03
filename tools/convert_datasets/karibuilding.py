# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import numpy as np

import json
import mmcv
# from cityscapesscripts.preparation.json2labelImg import json2labelImg

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert KARI Building Detection data to polygons.json')
    parser.add_argument('kari_building_path', help='KARI Building Detection data path')
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    kari_building_path = args.kari_building_path
    out_dir = args.out_dir if args.out_dir else kari_building_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(kari_building_path, args.gt_dir)

    data_types = ['train','test','val']
    type_count = {}
    for data_type in data_types:
        type_count[data_type] = {}

    json_files = []
    file_list = mmcv.scandir(gt_dir, '.json', recursive=True)
    for filename in file_list:
        json_file = osp.join(gt_dir, filename)
        
        # Read json_file
        if 'polygons' not in json_file:
            with open(json_file) as file:
                json_data = json.load(file)
                json_result = {}
                json_result['imgHeight'] = 1024
                json_result['imgWidth'] = 1024
                json_result['objects'] = []

                for object in json_data['features']:
                    type_name = object['properties']['type_name']
                    
                    if type_name == "소형 시설":
                        type_name = 'small building'
                    elif type_name == "아파트":
                        type_name = 'apartment'
                    elif type_name == "공장":
                        type_name = 'factory'
                    elif type_name == "중형 단독 시설":
                        type_name = 'medium building'
                    elif type_name == "대형 시설":
                        type_name = ''
                    elif type_name == "컨테이너 박스":
                        type_name = ''
                    elif type_name == "기타":
                        type_name = ''
                    else:
                        type_name = ''

                    for data_type in data_types:
                        if type_name not in type_count[data_type]:
                            type_count[data_type][type_name] = 0
                        elif data_type in json_file:
                            type_count[data_type][type_name] = type_count[data_type][type_name] + 1

                    coords = object['properties']['building_imcoords']
                    coords = np.fromstring(coords, dtype=float, sep=',')
                    polygon_data = []
                    for i in range(int(len(coords)/2)):
                        polygon_data.append([coords.item(i*2), coords.item(i*2+1)])
                    if len(polygon_data) >= 2 and type_name != "":
                        json_result['objects'].append({'label': type_name, 'polygon':polygon_data})
                gtfine_polygon_filename = json_file[:-5]+'_gtFine_polygons.json'
                with open(gtfine_polygon_filename, 'w') as outfile:
                    json.dump(json_result, outfile, indent=4)
    for data_type in data_types:
        print("Data Type : " + data_type)
        print(type_count[data_type])
if __name__ == '__main__':
    main()
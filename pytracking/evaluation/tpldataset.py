import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text


class TPLDataset(BaseDataset):
    """
    Temple Color 128 dataset

    Publication:
        Encoding Color Information for Visual Tracking: Algorithms and Benchmark
        P. Liang, E. Blasch, and H. Ling
        TIP, 2015
        http://www.dabi.temple.edu/~hbling/publication/TColor-128.pdf

    Download the dataset from http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html
    """
    def __init__(self, exclude_otb=False):
        """
        args:
            exclude_otb (bool) - If True, sequences overlapping with the OTB dataset are excluded
        """
        super().__init__()
        self.base_path = self.env_settings.tpl_path
        self.sequence_info_list = self._get_sequence_info_list(exclude_otb)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'tpl', ground_truth_rect[init_omit:,:])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self, exclude_otb=False):
        sequence_info_list = [
            {"name": "Skating2", "path": "Skating2/img", "startFrame": 1, "endFrame": 707, "nz": 4,
             "ext": "jpg", "anno_path": "Skating2/Skating2_gt.txt"},
            {"name": "Pool_ce3", "path": "Pool_ce3/img", "startFrame": 1, "endFrame": 124, "nz": 4,
             "ext": "jpg", "anno_path": "Pool_ce3/Pool_ce3_gt.txt"},
            {"name": "Microphone_ce1", "path": "Microphone_ce1/img", "startFrame": 1, "endFrame": 204, "nz": 4,
             "ext": "jpg", "anno_path": "Microphone_ce1/Microphone_ce1_gt.txt"},
            {"name": "Torus", "path": "Torus/img", "startFrame": 1, "endFrame": 264, "nz": 4, "ext": "jpg",
             "anno_path": "Torus/Torus_gt.txt"},
            {"name": "Lemming", "path": "Lemming/img", "startFrame": 1, "endFrame": 1336, "nz": 4, "ext": "jpg",
             "anno_path": "Lemming/Lemming_gt.txt"},
            {"name": "Eagle_ce", "path": "Eagle_ce/img", "startFrame": 1, "endFrame": 112, "nz": 4,
             "ext": "jpg", "anno_path": "Eagle_ce/Eagle_ce_gt.txt"},
            {"name": "Skating_ce2", "path": "Skating_ce2/img", "startFrame": 1, "endFrame": 497, "nz": 4,
             "ext": "jpg", "anno_path": "Skating_ce2/Skating_ce2_gt.txt"},
            {"name": "Yo_yos_ce3", "path": "Yo_yos_ce3/img", "startFrame": 1, "endFrame": 201, "nz": 4,
             "ext": "jpg", "anno_path": "Yo_yos_ce3/Yo-yos_ce3_gt.txt"},
            {"name": "Board", "path": "Board/img", "startFrame": 1, "endFrame": 598, "nz": 4, "ext": "jpg",
             "anno_path": "Board/Board_gt.txt"},
            {"name": "Tennis_ce3", "path": "Tennis_ce3/img", "startFrame": 1, "endFrame": 204, "nz": 4,
             "ext": "jpg", "anno_path": "Tennis_ce3/Tennis_ce3_gt.txt"},
            {"name": "SuperMario_ce", "path": "SuperMario_ce/img", "startFrame": 1, "endFrame": 146, "nz": 4,
             "ext": "jpg", "anno_path": "SuperMario_ce/SuperMario_ce_gt.txt"},
            {"name": "Yo_yos_ce1", "path": "Yo_yos_ce1/img", "startFrame": 1, "endFrame": 235, "nz": 4,
             "ext": "jpg", "anno_path": "Yo_yos_ce1/Yo-yos_ce1_gt.txt"},
            {"name": "Soccer", "path": "Soccer/img", "startFrame": 1, "endFrame": 392, "nz": 4, "ext": "jpg",
             "anno_path": "Soccer/Soccer_gt.txt"},
            {"name": "Fish_ce2", "path": "Fish_ce2/img", "startFrame": 1, "endFrame": 573, "nz": 4,
             "ext": "jpg", "anno_path": "Fish_ce2/Fish_ce2_gt.txt"},
            {"name": "Liquor", "path": "Liquor/img", "startFrame": 1, "endFrame": 1741, "nz": 4, "ext": "jpg",
             "anno_path": "Liquor/Liquor_gt.txt"},
            {"name": "Plane_ce2", "path": "Plane_ce2/img", "startFrame": 1, "endFrame": 653, "nz": 4,
             "ext": "jpg", "anno_path": "Plane_ce2/Plane_ce2_gt.txt"},
            {"name": "Couple", "path": "Couple/img", "startFrame": 1, "endFrame": 140, "nz": 4, "ext": "jpg",
             "anno_path": "Couple/Couple_gt.txt"},
            {"name": "Logo_ce", "path": "Logo_ce/img", "startFrame": 1, "endFrame": 610, "nz": 4, "ext": "jpg",
             "anno_path": "Logo_ce/Logo_ce_gt.txt"},
            {"name": "Hand_ce2", "path": "Hand_ce2/img", "startFrame": 1, "endFrame": 251, "nz": 4,
             "ext": "jpg", "anno_path": "Hand_ce2/Hand_ce2_gt.txt"},
            {"name": "Kite_ce2", "path": "Kite_ce2/img", "startFrame": 1, "endFrame": 658, "nz": 4,
             "ext": "jpg", "anno_path": "Kite_ce2/Kite_ce2_gt.txt"},
            {"name": "Walking", "path": "Walking/img", "startFrame": 1, "endFrame": 412, "nz": 4, "ext": "jpg",
             "anno_path": "Walking/Walking_gt.txt"},
            {"name": "David", "path": "David/img", "startFrame": 300, "endFrame": 770, "nz": 4, "ext": "jpg",
             "anno_path": "David/David_gt.txt"},
            {"name": "Boat_ce1", "path": "Boat_ce1/img", "startFrame": 1, "endFrame": 377, "nz": 4,
             "ext": "jpg", "anno_path": "Boat_ce1/Boat_ce1_gt.txt"},
            {"name": "Airport_ce", "path": "Airport_ce/img", "startFrame": 1, "endFrame": 148, "nz": 4,
             "ext": "jpg", "anno_path": "Airport_ce/Airport_ce_gt.txt"},
            {"name": "Tiger2", "path": "Tiger2/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg",
             "anno_path": "Tiger2/Tiger2_gt.txt"},
            {"name": "Suitcase_ce", "path": "Suitcase_ce/img", "startFrame": 1, "endFrame": 184, "nz": 4,
             "ext": "jpg", "anno_path": "Suitcase_ce/Suitcase_ce_gt.txt"},
            {"name": "TennisBall_ce", "path": "TennisBall_ce/img", "startFrame": 1, "endFrame": 288, "nz": 4,
             "ext": "jpg", "anno_path": "TennisBall_ce/TennisBall_ce_gt.txt"},
            {"name": "Singer_ce1", "path": "Singer_ce1/img", "startFrame": 1, "endFrame": 214, "nz": 4,
             "ext": "jpg", "anno_path": "Singer_ce1/Singer_ce1_gt.txt"},
            {"name": "Pool_ce2", "path": "Pool_ce2/img", "startFrame": 1, "endFrame": 133, "nz": 4,
             "ext": "jpg", "anno_path": "Pool_ce2/Pool_ce2_gt.txt"},
            {"name": "Surf_ce3", "path": "Surf_ce3/img", "startFrame": 1, "endFrame": 279, "nz": 4,
             "ext": "jpg", "anno_path": "Surf_ce3/Surf_ce3_gt.txt"},
            {"name": "Bird", "path": "Bird/img", "startFrame": 1, "endFrame": 99, "nz": 4, "ext": "jpg",
             "anno_path": "Bird/Bird_gt.txt"},
            {"name": "Crossing", "path": "Crossing/img", "startFrame": 1, "endFrame": 120, "nz": 4,
             "ext": "jpg", "anno_path": "Crossing/Crossing_gt.txt"},
            {"name": "Plate_ce1", "path": "Plate_ce1/img", "startFrame": 1, "endFrame": 142, "nz": 4,
             "ext": "jpg", "anno_path": "Plate_ce1/Plate_ce1_gt.txt"},
            {"name": "Cup", "path": "Cup/img", "startFrame": 1, "endFrame": 303, "nz": 4, "ext": "jpg",
             "anno_path": "Cup/Cup_gt.txt"},
            {"name": "Surf_ce2", "path": "Surf_ce2/img", "startFrame": 1, "endFrame": 391, "nz": 4,
             "ext": "jpg", "anno_path": "Surf_ce2/Surf_ce2_gt.txt"},
            {"name": "Busstation_ce2", "path": "Busstation_ce2/img", "startFrame": 6, "endFrame": 400, "nz": 4,
             "ext": "jpg", "anno_path": "Busstation_ce2/Busstation_ce2_gt.txt"},
            {"name": "Charger_ce", "path": "Charger_ce/img", "startFrame": 1, "endFrame": 298, "nz": 4,
             "ext": "jpg", "anno_path": "Charger_ce/Charger_ce_gt.txt"},
            {"name": "Pool_ce1", "path": "Pool_ce1/img", "startFrame": 1, "endFrame": 166, "nz": 4,
             "ext": "jpg", "anno_path": "Pool_ce1/Pool_ce1_gt.txt"},
            {"name": "MountainBike", "path": "MountainBike/img", "startFrame": 1, "endFrame": 228, "nz": 4,
             "ext": "jpg", "anno_path": "MountainBike/MountainBike_gt.txt"},
            {"name": "Guitar_ce1", "path": "Guitar_ce1/img", "startFrame": 1, "endFrame": 268, "nz": 4,
             "ext": "jpg", "anno_path": "Guitar_ce1/Guitar_ce1_gt.txt"},
            {"name": "Busstation_ce1", "path": "Busstation_ce1/img", "startFrame": 1, "endFrame": 363, "nz": 4,
             "ext": "jpg", "anno_path": "Busstation_ce1/Busstation_ce1_gt.txt"},
            {"name": "Diving", "path": "Diving/img", "startFrame": 1, "endFrame": 231, "nz": 4, "ext": "jpg",
             "anno_path": "Diving/Diving_gt.txt"},
            {"name": "Skating_ce1", "path": "Skating_ce1/img", "startFrame": 1, "endFrame": 409, "nz": 4,
             "ext": "jpg", "anno_path": "Skating_ce1/Skating_ce1_gt.txt"},
            {"name": "Hurdle_ce2", "path": "Hurdle_ce2/img", "startFrame": 27, "endFrame": 330, "nz": 4,
             "ext": "jpg", "anno_path": "Hurdle_ce2/Hurdle_ce2_gt.txt"},
            {"name": "Plate_ce2", "path": "Plate_ce2/img", "startFrame": 1, "endFrame": 181, "nz": 4,
             "ext": "jpg", "anno_path": "Plate_ce2/Plate_ce2_gt.txt"},
            {"name": "CarDark", "path": "CarDark/img", "startFrame": 1, "endFrame": 393, "nz": 4, "ext": "jpg",
             "anno_path": "CarDark/CarDark_gt.txt"},
            {"name": "Singer_ce2", "path": "Singer_ce2/img", "startFrame": 1, "endFrame": 999, "nz": 4,
             "ext": "jpg", "anno_path": "Singer_ce2/Singer_ce2_gt.txt"},
            {"name": "Shaking", "path": "Shaking/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg",
             "anno_path": "Shaking/Shaking_gt.txt"},
            {"name": "Iceskater", "path": "Iceskater/img", "startFrame": 1, "endFrame": 500, "nz": 4,
             "ext": "jpg", "anno_path": "Iceskater/Iceskater_gt.txt"},
            {"name": "Badminton_ce2", "path": "Badminton_ce2/img", "startFrame": 1, "endFrame": 705, "nz": 4,
             "ext": "jpg", "anno_path": "Badminton_ce2/Badminton_ce2_gt.txt"},
            {"name": "Spiderman_ce", "path": "Spiderman_ce/img", "startFrame": 1, "endFrame": 351, "nz": 4,
             "ext": "jpg", "anno_path": "Spiderman_ce/Spiderman_ce_gt.txt"},
            {"name": "Kite_ce1", "path": "Kite_ce1/img", "startFrame": 1, "endFrame": 484, "nz": 4,
             "ext": "jpg", "anno_path": "Kite_ce1/Kite_ce1_gt.txt"},
            {"name": "Skyjumping_ce", "path": "Skyjumping_ce/img", "startFrame": 1, "endFrame": 938, "nz": 4,
             "ext": "jpg", "anno_path": "Skyjumping_ce/Skyjumping_ce_gt.txt"},
            {"name": "Ball_ce1", "path": "Ball_ce1/img", "startFrame": 1, "endFrame": 391, "nz": 4,
             "ext": "jpg", "anno_path": "Ball_ce1/Ball_ce1_gt.txt"},
            {"name": "Yo_yos_ce2", "path": "Yo_yos_ce2/img", "startFrame": 1, "endFrame": 454, "nz": 4,
             "ext": "jpg", "anno_path": "Yo_yos_ce2/Yo-yos_ce2_gt.txt"},
            {"name": "Ironman", "path": "Ironman/img", "startFrame": 1, "endFrame": 166, "nz": 4, "ext": "jpg",
             "anno_path": "Ironman/Ironman_gt.txt"},
            {"name": "FaceOcc1", "path": "FaceOcc1/img", "startFrame": 1, "endFrame": 892, "nz": 4,
             "ext": "jpg", "anno_path": "FaceOcc1/FaceOcc1_gt.txt"},
            {"name": "Surf_ce1", "path": "Surf_ce1/img", "startFrame": 1, "endFrame": 404, "nz": 4,
             "ext": "jpg", "anno_path": "Surf_ce1/Surf_ce1_gt.txt"},
            {"name": "Ring_ce", "path": "Ring_ce/img", "startFrame": 1, "endFrame": 201, "nz": 4, "ext": "jpg",
             "anno_path": "Ring_ce/Ring_ce_gt.txt"},
            {"name": "Surf_ce4", "path": "Surf_ce4/img", "startFrame": 1, "endFrame": 135, "nz": 4,
             "ext": "jpg", "anno_path": "Surf_ce4/Surf_ce4_gt.txt"},
            {"name": "Ball_ce4", "path": "Ball_ce4/img", "startFrame": 1, "endFrame": 538, "nz": 4,
             "ext": "jpg", "anno_path": "Ball_ce4/Ball_ce4_gt.txt"},
            {"name": "Bikeshow_ce", "path": "Bikeshow_ce/img", "startFrame": 1, "endFrame": 361, "nz": 4,
             "ext": "jpg", "anno_path": "Bikeshow_ce/Bikeshow_ce_gt.txt"},
            {"name": "Kobe_ce", "path": "Kobe_ce/img", "startFrame": 1, "endFrame": 582, "nz": 4, "ext": "jpg",
             "anno_path": "Kobe_ce/Kobe_ce_gt.txt"},
            {"name": "Tiger1", "path": "Tiger1/img", "startFrame": 1, "endFrame": 354, "nz": 4, "ext": "jpg",
             "anno_path": "Tiger1/Tiger1_gt.txt"},
            {"name": "Skiing", "path": "Skiing/img", "startFrame": 1, "endFrame": 81, "nz": 4, "ext": "jpg",
             "anno_path": "Skiing/Skiing_gt.txt"},
            {"name": "Tennis_ce1", "path": "Tennis_ce1/img", "startFrame": 1, "endFrame": 454, "nz": 4,
             "ext": "jpg", "anno_path": "Tennis_ce1/Tennis_ce1_gt.txt"},
            {"name": "Carchasing_ce4", "path": "Carchasing_ce4/img", "startFrame": 1, "endFrame": 442, "nz": 4,
             "ext": "jpg", "anno_path": "Carchasing_ce4/Carchasing_ce4_gt.txt"},
            {"name": "Walking2", "path": "Walking2/img", "startFrame": 1, "endFrame": 500, "nz": 4,
             "ext": "jpg", "anno_path": "Walking2/Walking2_gt.txt"},
            {"name": "Sailor_ce", "path": "Sailor_ce/img", "startFrame": 1, "endFrame": 402, "nz": 4,
             "ext": "jpg", "anno_path": "Sailor_ce/Sailor_ce_gt.txt"},
            {"name": "Railwaystation_ce", "path": "Railwaystation_ce/img", "startFrame": 1, "endFrame": 413,
             "nz": 4, "ext": "jpg", "anno_path": "Railwaystation_ce/Railwaystation_ce_gt.txt"},
            {"name": "Bee_ce", "path": "Bee_ce/img", "startFrame": 1, "endFrame": 90, "nz": 4, "ext": "jpg",
             "anno_path": "Bee_ce/Bee_ce_gt.txt"},
            {"name": "Girl", "path": "Girl/img", "startFrame": 1, "endFrame": 500, "nz": 4, "ext": "jpg",
             "anno_path": "Girl/Girl_gt.txt"},
            {"name": "Subway", "path": "Subway/img", "startFrame": 1, "endFrame": 175, "nz": 4, "ext": "jpg",
             "anno_path": "Subway/Subway_gt.txt"},
            {"name": "David3", "path": "David3/img", "startFrame": 1, "endFrame": 252, "nz": 4, "ext": "jpg",
             "anno_path": "David3/David3_gt.txt"},
            {"name": "Electricalbike_ce", "path": "Electricalbike_ce/img", "startFrame": 1, "endFrame": 818,
             "nz": 4, "ext": "jpg", "anno_path": "Electricalbike_ce/Electricalbike_ce_gt.txt"},
            {"name": "Michaeljackson_ce", "path": "Michaeljackson_ce/img", "startFrame": 1, "endFrame": 393,
             "nz": 4, "ext": "jpg", "anno_path": "Michaeljackson_ce/Michaeljackson_ce_gt.txt"},
            {"name": "Woman", "path": "Woman/img", "startFrame": 1, "endFrame": 597, "nz": 4, "ext": "jpg",
             "anno_path": "Woman/Woman_gt.txt"},
            {"name": "TableTennis_ce", "path": "TableTennis_ce/img", "startFrame": 1, "endFrame": 198, "nz": 4,
             "ext": "jpg", "anno_path": "TableTennis_ce/TableTennis_ce_gt.txt"},
            {"name": "Motorbike_ce", "path": "Motorbike_ce/img", "startFrame": 1, "endFrame": 563, "nz": 4,
             "ext": "jpg", "anno_path": "Motorbike_ce/Motorbike_ce_gt.txt"},
            {"name": "Baby_ce", "path": "Baby_ce/img", "startFrame": 1, "endFrame": 296, "nz": 4, "ext": "jpg",
             "anno_path": "Baby_ce/Baby_ce_gt.txt"},
            {"name": "Gym", "path": "Gym/img", "startFrame": 1, "endFrame": 766, "nz": 4, "ext": "jpg",
             "anno_path": "Gym/Gym_gt.txt"},
            {"name": "Matrix", "path": "Matrix/img", "startFrame": 1, "endFrame": 100, "nz": 4, "ext": "jpg",
             "anno_path": "Matrix/Matrix_gt.txt"},
            {"name": "Kite_ce3", "path": "Kite_ce3/img", "startFrame": 1, "endFrame": 528, "nz": 4,
             "ext": "jpg", "anno_path": "Kite_ce3/Kite_ce3_gt.txt"},
            {"name": "Fish_ce1", "path": "Fish_ce1/img", "startFrame": 1, "endFrame": 401, "nz": 4,
             "ext": "jpg", "anno_path": "Fish_ce1/Fish_ce1_gt.txt"},
            {"name": "Hand_ce1", "path": "Hand_ce1/img", "startFrame": 1, "endFrame": 401, "nz": 4,
             "ext": "jpg", "anno_path": "Hand_ce1/Hand_ce1_gt.txt"},
            {"name": "Doll", "path": "Doll/img", "startFrame": 1, "endFrame": 3872, "nz": 4, "ext": "jpg",
             "anno_path": "Doll/Doll_gt.txt"},
            {"name": "Carchasing_ce3", "path": "Carchasing_ce3/img", "startFrame": 1, "endFrame": 572, "nz": 4,
             "ext": "jpg", "anno_path": "Carchasing_ce3/Carchasing_ce3_gt.txt"},
            {"name": "Thunder_ce", "path": "Thunder_ce/img", "startFrame": 1, "endFrame": 375, "nz": 4,
             "ext": "jpg", "anno_path": "Thunder_ce/Thunder_ce_gt.txt"},
            {"name": "Singer2", "path": "Singer2/img", "startFrame": 1, "endFrame": 366, "nz": 4, "ext": "jpg",
             "anno_path": "Singer2/Singer2_gt.txt"},
            {"name": "Basketball", "path": "Basketball/img", "startFrame": 1, "endFrame": 725, "nz": 4,
             "ext": "jpg", "anno_path": "Basketball/Basketball_gt.txt"},
            {"name": "Hand", "path": "Hand/img", "startFrame": 1, "endFrame": 244, "nz": 4, "ext": "jpg",
             "anno_path": "Hand/Hand_gt.txt"},
            {"name": "Cup_ce", "path": "Cup_ce/img", "startFrame": 1, "endFrame": 338, "nz": 4, "ext": "jpg",
             "anno_path": "Cup_ce/Cup_ce_gt.txt"},
            {"name": "MotorRolling", "path": "MotorRolling/img", "startFrame": 1, "endFrame": 164, "nz": 4,
             "ext": "jpg", "anno_path": "MotorRolling/MotorRolling_gt.txt"},
            {"name": "Boat_ce2", "path": "Boat_ce2/img", "startFrame": 1, "endFrame": 412, "nz": 4,
             "ext": "jpg", "anno_path": "Boat_ce2/Boat_ce2_gt.txt"},
            {"name": "CarScale", "path": "CarScale/img", "startFrame": 1, "endFrame": 252, "nz": 4,
             "ext": "jpg", "anno_path": "CarScale/CarScale_gt.txt"},
            {"name": "Sunshade", "path": "Sunshade/img", "startFrame": 1, "endFrame": 172, "nz": 4,
             "ext": "jpg", "anno_path": "Sunshade/Sunshade_gt.txt"},
            {"name": "Football1", "path": "Football1/img", "startFrame": 1, "endFrame": 74, "nz": 4,
             "ext": "jpg", "anno_path": "Football1/Football1_gt.txt"},
            {"name": "Singer1", "path": "Singer1/img", "startFrame": 1, "endFrame": 351, "nz": 4, "ext": "jpg",
             "anno_path": "Singer1/Singer1_gt.txt"},
            {"name": "Hurdle_ce1", "path": "Hurdle_ce1/img", "startFrame": 1, "endFrame": 300, "nz": 4,
             "ext": "jpg", "anno_path": "Hurdle_ce1/Hurdle_ce1_gt.txt"},
            {"name": "Basketball_ce3", "path": "Basketball_ce3/img", "startFrame": 1, "endFrame": 441, "nz": 4,
             "ext": "jpg", "anno_path": "Basketball_ce3/Basketball_ce3_gt.txt"},
            {"name": "Toyplane_ce", "path": "Toyplane_ce/img", "startFrame": 1, "endFrame": 405, "nz": 4,
             "ext": "jpg", "anno_path": "Toyplane_ce/Toyplane_ce_gt.txt"},
            {"name": "Skating1", "path": "Skating1/img", "startFrame": 1, "endFrame": 400, "nz": 4,
             "ext": "jpg", "anno_path": "Skating1/Skating1_gt.txt"},
            {"name": "Juice", "path": "Juice/img", "startFrame": 1, "endFrame": 404, "nz": 4, "ext": "jpg",
             "anno_path": "Juice/Juice_gt.txt"},
            {"name": "Biker", "path": "Biker/img", "startFrame": 1, "endFrame": 180, "nz": 4, "ext": "jpg",
             "anno_path": "Biker/Biker_gt.txt"},
            {"name": "Boy", "path": "Boy/img", "startFrame": 1, "endFrame": 602, "nz": 4, "ext": "jpg",
             "anno_path": "Boy/Boy_gt.txt"},
            {"name": "Jogging1", "path": "Jogging1/img", "startFrame": 1, "endFrame": 307, "nz": 4,
             "ext": "jpg", "anno_path": "Jogging1/Jogging1_gt.txt"},
            {"name": "Deer", "path": "Deer/img", "startFrame": 1, "endFrame": 71, "nz": 4, "ext": "jpg",
             "anno_path": "Deer/Deer_gt.txt"},
            {"name": "Panda", "path": "Panda/img", "startFrame": 1, "endFrame": 241, "nz": 4, "ext": "jpg",
             "anno_path": "Panda/Panda_gt.txt"},
            {"name": "Coke", "path": "Coke/img", "startFrame": 1, "endFrame": 291, "nz": 4, "ext": "jpg",
             "anno_path": "Coke/Coke_gt.txt"},
            {"name": "Carchasing_ce1", "path": "Carchasing_ce1/img", "startFrame": 1, "endFrame": 501, "nz": 4,
             "ext": "jpg", "anno_path": "Carchasing_ce1/Carchasing_ce1_gt.txt"},
            {"name": "Badminton_ce1", "path": "Badminton_ce1/img", "startFrame": 1, "endFrame": 579, "nz": 4,
             "ext": "jpg", "anno_path": "Badminton_ce1/Badminton_ce1_gt.txt"},
            {"name": "Trellis", "path": "Trellis/img", "startFrame": 1, "endFrame": 569, "nz": 4, "ext": "jpg",
             "anno_path": "Trellis/Trellis_gt.txt"},
            {"name": "Face_ce2", "path": "Face_ce2/img", "startFrame": 1, "endFrame": 148, "nz": 4,
             "ext": "jpg", "anno_path": "Face_ce2/Face_ce2_gt.txt"},
            {"name": "Ball_ce2", "path": "Ball_ce2/img", "startFrame": 1, "endFrame": 603, "nz": 4,
             "ext": "jpg", "anno_path": "Ball_ce2/Ball_ce2_gt.txt"},
            {"name": "Skiing_ce", "path": "Skiing_ce/img", "startFrame": 1, "endFrame": 511, "nz": 4,
             "ext": "jpg", "anno_path": "Skiing_ce/Skiing_ce_gt.txt"},
            {"name": "Jogging2", "path": "Jogging2/img", "startFrame": 1, "endFrame": 307, "nz": 4,
             "ext": "jpg", "anno_path": "Jogging2/Jogging2_gt.txt"},
            {"name": "Bike_ce1", "path": "Bike_ce1/img", "startFrame": 1, "endFrame": 801, "nz": 4,
             "ext": "jpg", "anno_path": "Bike_ce1/Bike_ce1_gt.txt"},
            {"name": "Bike_ce2", "path": "Bike_ce2/img", "startFrame": 1, "endFrame": 812, "nz": 4,
             "ext": "jpg", "anno_path": "Bike_ce2/Bike_ce2_gt.txt"},
            {"name": "Ball_ce3", "path": "Ball_ce3/img", "startFrame": 1, "endFrame": 273, "nz": 4,
             "ext": "jpg", "anno_path": "Ball_ce3/Ball_ce3_gt.txt"},
            {"name": "Girlmov", "path": "Girlmov/img", "startFrame": 1, "endFrame": 1500, "nz": 4, "ext": "jpg",
             "anno_path": "Girlmov/Girlmov_gt.txt"},
            {"name": "Bolt", "path": "Bolt/img", "startFrame": 1, "endFrame": 350, "nz": 4, "ext": "jpg",
             "anno_path": "Bolt/Bolt_gt.txt"},
            {"name": "Basketball_ce2", "path": "Basketball_ce2/img", "startFrame": 1, "endFrame": 455, "nz": 4,
             "ext": "jpg", "anno_path": "Basketball_ce2/Basketball_ce2_gt.txt"},
            {"name": "Bicycle", "path": "Bicycle/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg",
             "anno_path": "Bicycle/Bicycle_gt.txt"},
            {"name": "Face_ce", "path": "Face_ce/img", "startFrame": 1, "endFrame": 620, "nz": 4, "ext": "jpg",
             "anno_path": "Face_ce/Face_ce_gt.txt"},
            {"name": "Basketball_ce1", "path": "Basketball_ce1/img", "startFrame": 1, "endFrame": 496, "nz": 4,
             "ext": "jpg", "anno_path": "Basketball_ce1/Basketball_ce1_gt.txt"},
            {"name": "Messi_ce", "path": "Messi_ce/img", "startFrame": 1, "endFrame": 272, "nz": 4,
             "ext": "jpg", "anno_path": "Messi_ce/Messi_ce_gt.txt"},
            {"name": "Tennis_ce2", "path": "Tennis_ce2/img", "startFrame": 1, "endFrame": 305, "nz": 4,
             "ext": "jpg", "anno_path": "Tennis_ce2/Tennis_ce2_gt.txt"},
            {"name": "Microphone_ce2", "path": "Microphone_ce2/img", "startFrame": 1, "endFrame": 103, "nz": 4,
             "ext": "jpg", "anno_path": "Microphone_ce2/Microphone_ce2_gt.txt"},
            {"name": "Guitar_ce2", "path": "Guitar_ce2/img", "startFrame": 1, "endFrame": 313, "nz": 4,
             "ext": "jpg", "anno_path": "Guitar_ce2/Guitar_ce2_gt.txt"}

        ]

        otb_sequences = ['tpl_Skating2', 'tpl_Lemming', 'tpl_Board', 'tpl_Soccer', 'tpl_Liquor', 'tpl_Couple', 'tpl_Walking', 'tpl_David', 'tpl_Tiger2', 'tpl_Bird', 'tpl_Crossing', 'tpl_MountainBike',
                         'tpl_Diving', 'tpl_CarDark', 'tpl_Shaking', 'tpl_Ironman', 'tpl_FaceOcc1', 'tpl_Tiger1', 'tpl_Skiing', 'tpl_Walking2', 'tpl_Girl', 'tpl_Girlmov', 'tpl_Subway', 'tpl_David3', 'tpl_Woman',
                         'tpl_Gym', 'tpl_Matrix', 'tpl_Doll', 'tpl_Singer2', 'tpl_Basketball', 'tpl_MotorRolling', 'tpl_CarScale', 'tpl_Football1', 'tpl_Singer1', 'tpl_Skating1', 'tpl_Biker',
                         'tpl_Boy', 'tpl_Jogging1', 'tpl_Deer', 'tpl_Panda', 'tpl_Coke', 'tpl_Trellis', 'tpl_Jogging2', 'tpl_Bolt', ]
        if exclude_otb:
            sequence_info_list_nootb = []
            for seq in sequence_info_list:
                if seq['name'] not in otb_sequences:
                    sequence_info_list_nootb.append(seq)

            sequence_info_list = sequence_info_list_nootb

        return sequence_info_list

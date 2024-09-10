import os
import cv2
from glob import glob
import numpy as np

image_format = 'png'


class ItemIdGenerator(object):
    
    def __init__(self):
        self.index = 0
    
    
    def next(self) -> int:
        self.index += 1
        return self.index


item_id_gen = ItemIdGenerator()


class PointSet(object):

    def __init__(self):
        self.points = []
    
    def __len__(self):
        return len(self.points)
    
    def __iter__(self):
        return iter(self.points)
    
    def add(self, x: float, y: float, label: int):
        self.points.append((x, y, label))

    def pop(self):
        if not self.points:
            return
        self.points.pop()

    def clear(self):
        if not self.points:
            return
        self.points.clear()
    
    def __str__(self):
        return f'PointSet({self.x}, {self.y}, {self.label})'


class ImageFrame(object):

    def __init__(self, frame_data: np.ndarray, origin_index: int):
        '''
            This object represents a image frame extraced from video, with the points for segmentation included.
            ::params::
                frame_data: image frame data
                origin_index: frame index of the original video;
        '''
        self.frame_data = frame_data
        self.preview_data = None
        self.origin_index = origin_index
        self.point_set = PointSet()
    
       
    def set_preview(self, preview: np.ndarray):
        self.preview_data = preview


    def get_preview(self) -> np.ndarray:
        return self.preview_data
    
    
    def add(self, x: float, y: float, label: int):
        '''
           Add one segment point.
           ::params::
               x: x position on the frame;
               y: y position on the frame;
               label: 0 = background, 1 = foreground
        '''
        self.point_set.add(x, y, label)
        self.set_preview(None)


    def pop(self):
        '''
            Pop up the latests added point.
            Do nothing if no point exists.
        '''
        self.point_set.pop()
        self.set_preview(None)


    def clear(self):
        '''
            Clear all points.
            Do nothing if no point exists.
        '''
        self.point_set.clear()
        self.set_preview(None)


    def __str__(self):
        return f'ImageFrame(origin_index={self.origin_index},frame_data=[{len(self.frame_data)}])'


class SegmentItem(object):

    @staticmethod
    def create(name: str):
        return SegmentItem(name, item_id_gen.next())

    def __init__(self, name: str, item_id: int):
        '''
            This object represents one specific item to be segmented out.
            ::params::
                name: human-readable name for identify the item; for display purpose only, has no pragmatic effect;
                item_id: global unique item id; used for sam2 inteference;
        '''
        self.name = name
        self.frames = []
        self.item_id = item_id
        self.current_index = 0
        if not self.name:
            self.name = f'Item{item_id}'

    def __len__(self) -> int:
        return len(self.frames)


    def __iter__(self) -> int:
        return iter(self.frames)

    
    def add_frame(self, frame: ImageFrame):
        '''
            Add a image frame into this item. Will set the new frame as current frame as well.
            ::params::
                frame: the frame to be added;
        '''
        if not frame:
            return
        if not self.frames:
            self.frames.append(frame)
            return
        if frame.origin_index in set([x.origin_index for x in self.frames]):
            return
        
        found = False
        for i, f in enumerate(self.frames):
            if f.origin_index > frame.origin_index:
                self.frames.insert(i, frame)
                found = True
                self.current_index = i
                break
        if not found:
            self.current_index = len(self.frames)
            self.frames.append(frame)

    def select_frame(self, index):
        '''
            Select certain frame as current frame
            ::params::
                index: the frame index of current frame;
        '''
        if index >= len(self.frames):
            return
        self.current_index = index

    def current_frame(self):
        '''
            Return current selected frame.
        '''
        if not self.frames:
            return None
        return self.frames[self.current_index]

    def remove_current(self) -> ImageFrame:
        '''
            Remove current frame and return it's next one if exists else the last one;
        '''
        if not self.frames:
            return None
        if len(self.frames) == 1:
            self.frames.clear()
            return None
        self.frames.pop(self.current_index)
        if self.current_index >= len(self.frames):
            self.current_index = len(self.frames) - 1
        return self.current_frame()

    def __str__(self):
        return f'SegmentItem(name={self.name},item_id={self.item_id},cur_idx={self.current_index}, frames=[{len(self.frames)}])'


class SegmentItemContainer(object):
    _instance = None
    
    @staticmethod
    def instance():
        if SegmentItemContainer._instance:
            return SegmentItemContainer._instance
        SegmentItemContainer._instance = SegmentItemContainer()
        return SegmentItemContainer._instance

    def __init__(self):
        '''
            This object managers all the items in one specific video.
        '''
        self.items_list = []
        self.current_index = 0
    

    def __iter__(self):
        return iter(self.items_list)


    def __len__(self) -> int:
        return len(self.items_list)
    
    
    def exists(self, name: str) -> bool:
        '''
            Test whether name has exists.
        '''
        return name in set([x.name for x in self.items_list])
    
    
    def add_item(self, item: SegmentItem):
        '''
            Add one item to this video. Will set the new item as current item as well.
            ::params::
                item: the item to be added.
        '''
        result = None
        if not self.items_list:
            self.items_list.append(item)
        elif item in set([it.item_id for it in self.items_list]):
            return
        else:
            found = False
            for idx, it in enumerate(self.items_list):
                if it.item_id > item.item_id:
                    self.items_list.insert(idx, it)
                    found = True
                    self.current_index = idx
                    break
            if not found:
                self.current_index = len(self.items_list)
                self.items_list.append(item)

    def select_item(self, index: int):
        '''
            Select certain item as current item
            ::params::
                index: the item index of current item;
        '''
        if index >= len(self.items_list):
            return
        self.current_index = index



    def select_by_item_id(self, item_id):
        '''
            Select certain item as current item by item id
            ::params::
                item_id: the item id to be selected; do nothing if none matches;
        '''
        for i, it in enumerate(self.items_list):
            if it.item_id == item_id:
                self.current_index = i
                break

    
    def select_by_item_name(self, item_name):
        '''
            Select certain item as current item by item id
            ::params::
                item_name: the item name to be selected; do nothing if none matches;
        '''
        for i, it in enumerate(self.items_list):
            if it.name == item_name:
                self.current_index = i
                break

    def current_item(self) -> SegmentItem:
        '''
            Return current selected item.
        '''
        if not self.items_list:
            return None
        if self.current_index >= len(self.items_list):
            return None
        return self.items_list[self.current_index]
    
    def remove_current(self) -> SegmentItem:
        '''
            Remove current item and return it's next one if exists else the last one;
        '''
        if not self.items_list:
            return None
        if len(self.items_list) == 1:
            self.items_list.clear()
            return None
        self.items_list.pop(self.current_index)
        if self.current_index >= len(self.items_list):
            self.current_index = len(self.items_list) - 1
        return self.current_item()

    def clear(self):
        self.items_list = []
        self.current_index = 0
    
    
    def __str__(self):
        return f'SegmentItemContainer(current_index={self.current_index},item_list=[{len(self.items_list)}])'


def count_video_frame_total(video_path : str) -> int:
    '''
        Calculate the frame count for a specific video file.
    '''
    video_capture = cv2.VideoCapture(video_path)
    if video_capture.isOpened():
        video_frame_total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_capture.release()
        return video_frame_total
    return 0


def get_video_frame(video_path : str, frame_number : int = 0) -> np.ndarray:
    '''
        Get frame data at specific frame number
    '''
    video_capture = cv2.VideoCapture(video_path)
    if video_capture.isOpened():
        frame_total = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1))
        has_vision_frame, vision_frame = video_capture.read()
        video_capture.release()
        if has_vision_frame:
            return cv2.cvtColor(vision_frame, cv2.COLOR_BGR2RGB)
    return None
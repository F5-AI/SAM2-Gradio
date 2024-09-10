import os
import cv2
import torch
import numpy as np
import gradio as gr
from image_segment import image_inference
from video_segment import video_interfrence, InterferenceFrame
from video_process import count_video_frame_total, get_video_frame, SegmentItemContainer, SegmentItem, ImageFrame
from glob import glob


wdir = os.path.dirname(__file__)
os.chdir(wdir)
# points color and marker
colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]


image_examples = sorted(list(glob(os.path.join(os.curdir, 'images', '*.jpg'))))
video_examples = sorted(list(glob(os.path.join(os.curdir, 'images', '*.mp4'))))



# ---- Video Global Variables ----
current_origin_frame = None
# ---- End ----


# ---- Video Global Variables ----
# current video file
current_video_file = video_examples[0]
# item container
item_container = SegmentItemContainer.instance()
# ---- End ----


def add_mark(frame):
    if frame is None:
        return None
    result = frame.frame_data.copy()
    marker_size = 25
    marker_thickness = 3
    marker_default_width = 1200
    width = result.shape[0]
    ratio = width / marker_default_width
    marker_final_size = int(marker_size * ratio)
    if marker_final_size < 3:
        marker_final_size = 3
    marker_final_thickness = int(marker_thickness * ratio)
    if marker_final_thickness < 2:
        marker_final_thickness = 2
    for (x, y, label) in frame.point_set:
        cv2.drawMarker(result, (x, y), colors[label], markerType=markers[label], markerSize=marker_final_size, thickness=marker_final_thickness)
    return result


def process_origin_image(img):
    global current_origin_frame
    current_origin_frame = ImageFrame(img, 0)
    return [None, None, '前景']


def new_existing_items_dropdown(choices = []):
    return gr.Dropdown(label = '选择物品', info = '输入新物品称名->回车加入新物品，或下拉选择物品', choices = choices, type = 'value', allow_custom_value=True)


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
            '''# Gradio-WebUI for Segment Anything Model 2 (SAM 2) 🚀'''
        )
    with gr.Row():
        device = gr.Dropdown(choices = ['cuda', 'cpu'], type='value', value='cuda', label='选择设备', visible = False)

    with gr.Tab(label='图像分割'):
    
        with gr.Row(equal_height=True):
            with gr.Column():
                input_image = gr.Image(type="numpy", label='待分割图片', format='png', image_mode='RGB')
            with gr.Column():
                with gr.Tab(label='叠加背景'):
                    output_image = gr.Image(type='numpy', format='png', image_mode='RGB')
                with gr.Tab(label='仅物品'):
                    output_mask = gr.Image(type='numpy', format='png', image_mode='RGBA')
        with gr.Row(equal_height = True):
            with gr.Column():
                with gr.Row():
                    gr.Markdown('选择样例图片或上传图片。')
                    undo_point_btn = gr.Button('撤销标记点')
                    remove_points_btn = gr.Button('清除标记点')
                
                bg_radio = gr.Radio(['前景', '背景'], label='标记点类型')

                image_ex = gr.Examples(
                    examples=image_examples,
                    examples_per_page=20,
                    inputs=input_image,
                    fn=process_origin_image,
                    outputs=[output_image, output_mask, bg_radio],
                    run_on_click = True
                )
                image_commit_btn = gr.Button("物品分割")
            with gr.Column():
                gr.Image(visible = False)


    with gr.Tab(label='视频分割'):
        with gr.Row(equal_height=True):
            with gr.Column():
                with gr.Row():
                    with gr.Column(variant = 'panel'):
                        gr.Markdown('### 步骤1:&nbsp;&nbsp;选择/上传原始视频')
                        input_video = gr.Video(label='原始视频', value=video_examples[0])
                        gr.Examples(
                            label = '样例视频',
                            examples=video_examples,
                            inputs=input_video,
                            examples_per_page=20,
                        )
                with gr.Row():
                    with gr.Column(variant = 'panel'):
                        gr.Markdown('### 步骤3:&nbsp;&nbsp;提交标记结果进行视频分割')
                        output_video = gr.Video(format='mp4', label='输出视频', interactive=False)
                        button_video = gr.Button(value = '视频分割')

            with gr.Column():
                gr.Markdown('### 步骤2:&nbsp;&nbsp;编辑分割物品')
                with gr.Row():
                    with gr.Column(variant='panel'):
                        maximum = count_video_frame_total(current_video_file)
                        origin_frame = gr.Image(label='原始帧预览', type='numpy', interactive=False,value=get_video_frame(current_video_file, 0))
                        origin_slider = gr.Slider(label='选择原视频帧', maximum = maximum, value = 0, step=1)

                with gr.Row():
                    with gr.Column(variant='panel'):
                        gr.Markdown('拖动上方的滑块选择包含目标物品的原始帧，从“选择物品”下拉框中输入新物品或选择已有物品，点击“加入物品”加入到“物品帧预览”中。')
                    with gr.Column(variant='panel'):
                        existing_items = new_existing_items_dropdown()
                        existing_item_btn = gr.Button('加入物品')

                with gr.Row(equal_height=True):
                    with gr.Column(variant='panel'):

                        with gr.Row(equal_height=True): 
                            with gr.Column(variant='panel'):
                                item_frame_preview = gr.Image(label='物品帧预览', interactive = False, sources=[])
                                origin_frame_preview = gr.Image(label='原始物品帧预览', interactive = False, sources=[], visible = False)
                                with gr.Row(equal_height=True):
                                    item_frame_slider = gr.Slider(label='选择物品帧', scale=5)
                                    item_origin_frame = gr.Number(label='原始帧序号', value=0, scale=1, min_width=10, interactive=False)
                                video_mark_radio = gr.Radio(label='标记类型', choices = ['前景', '背景'], value=0, type='index', interactive=True)
                                with gr.Row():
                                    gr.Markdown('撤销最近添加的标记点')
                                    undo_vedio_button = gr.Button(value='撤销标记')

                        with gr.Row(equal_height=True): 
                            with gr.Column(variant='panel'):
                                item_seg_preview = gr.Image(label='物品分割预览', interactive = False)
                                with gr.Row():
                                    gr.Markdown('点击查看分割预览结果')
                                    item_seg_btn = gr.Button(value='生成预览')

                        with gr.Tab(label='当前物品'):
                            with gr.Row():
                                item_id = gr.Number(label='物品ID', value=10, scale=1, min_width=10, interactive=False)
                                item_name = gr.TextArea(label='物品名称', value='', lines=1, max_lines=1, scale=1, min_width=20, interactive=False)
                            with gr.Row():
                                item_delete_button = gr.Button('删除物品')

                        with gr.Row(equal_height=True):
                            def update_current_item(it_id, it_name):
                                global item_container
                                item_container.select_by_item_id(it_id)
                                cur_item = item_container.current_item()
                                cur_frame = None
                                if cur_item is not None:
                                    cur_frame = cur_item.current_frame()
                                frame_preview = None
                                frame_data = None
                                if cur_frame and cur_frame.frame_data is not None:
                                    frame_data = cur_frame.frame_data.copy()
                                if cur_frame:
                                    frame_preview = gr.Image(label='物品帧预览', interactive = True, value=add_mark(cur_frame), sources=[])
                                else:
                                    frame_preview = gr.Image(label='物品帧预览', interactive = False)
                                    
                                frame_slider = None
                                if cur_item:
                                    maximum = len(cur_item) - 1
                                    if maximum == 0:
                                        maximum = 1
                                    frame_slider = gr.Slider(value=cur_item.current_index, minimum = 0, maximum = maximum, label='选择物品帧', scale=5, interactive = True)
                                else:
                                    frame_slider = gr.Slider(value=0, minimum = 0, maximum = 1, label='选择物品帧', scale=5, interactive = False)
                                origin_frame = 0
                                if cur_frame:
                                    origin_frame = cur_frame.origin_index

                                seg_preview = None
                                if cur_frame:
                                    seg_preview = cur_frame.preview_data
                                
                                origin_frame_data = None
                                if cur_frame and cur_frame.frame_data is not None:
                                    origin_frame_data = cur_frame.frame_data

                                return [frame_preview, origin_frame_data, frame_slider, origin_frame, seg_preview, it_id, it_name]
                            all_preview_image_widgets = [item_frame_preview, origin_frame_preview, item_frame_slider, item_origin_frame, item_seg_preview, item_id, item_name]
                            all_items_ex = gr.Examples(
                                label='选择标记物品',
                                examples = [[-1, '']],
                                inputs = [item_id, item_name],
                                fn = update_current_item,
                                run_on_click = True,
                                outputs = all_preview_image_widgets,
                            )


    
    
    # 图片分割事件处理函数

    gr.Image.input(input_image, process_origin_image, input_image, [output_image, output_mask, bg_radio])
    

    def add_mark_point(point_type, event: gr.SelectData):
        global current_origin_frame
        label = 1
        if point_type == '前景':
            label = 1
        elif point_type == '背景':
            label = 0
        
        if current_origin_frame is None:
            return None
        
        current_origin_frame.add(*event.index, label)
        return add_mark(current_origin_frame)

    gr.Image.select(input_image, add_mark_point, inputs = [bg_radio], outputs = [input_image])


    def undo_last_point():
        global current_origin_frame
        if current_origin_frame is None:
            return None
        current_origin_frame.pop()
        return add_mark(current_origin_frame)
    gr.Button.click(undo_point_btn, undo_last_point, inputs=None, outputs=input_image)


    def remove_all_points():
        global current_origin_frame
        if current_origin_frame is None:
            return None
        current_origin_frame.clear()
        return current_origin_frame.frame_data, '前景'

    gr.Button.click(remove_points_btn, remove_all_points, inputs=None, outputs=[input_image, bg_radio])

    
    def do_image_interference(device):
        global current_origin_frame
        if current_origin_frame is None:
            gr.Warning('请先选择图片', duration = 3)
            return None, None
        points = []
        if current_origin_frame is not None:
            for x, y, label in current_origin_frame.point_set:
                points.append(((x, y), label))
        
        return image_inference(device, current_origin_frame.frame_data, points)

    gr.Button.click(image_commit_btn, do_image_interference, inputs=[device], outputs=[output_image, output_mask])
    
    # 视频分割事件处理函数


    def change_video(path):
        global current_video_file
        global item_container
        if path != current_video_file:
            item_container.clear()
        current_video_file = path
        maximum = count_video_frame_total(current_video_file)
        slider = gr.Slider(minimum=0, maximum=maximum, label='原视频帧预览', value=0, step=1, interactive=True)
        return get_video_frame(current_video_file, 0), slider, *update_all_preview_widgets(True)


    def change_origin_preview(value):
        global current_video_file
        return get_video_frame(current_video_file, value - 1)
                

    def update_all_preview_widgets(update_all):
        global item_container
        results = []
        cur_item = item_container.current_item()
        
        if update_all:
            all_items = [(item.name, item.item_id) for item in item_container]
            item_names =  new_existing_items_dropdown(all_items)
            example_all_items = [(item[1], item[0]) for item in all_items]
            results = [item_names, gr.Dataset(samples = example_all_items)]

        if cur_item is None:
            slider = gr.Slider(value=0, minimum = 0, maximum = 1, label='选择物品帧', scale=5, interactive = False)
            results.extend([None, None, slider, 0, None, 0, ''])
        else:
            results.extend(update_current_item(cur_item.item_id, cur_item.name))
        return results


    def attach_existing_item(item_name, frame_data, frame_index):
        if item_name is None:
            gr.Warning(f'没有选中的物品, 请先添加物品并选择', duration = 3)
            return update_all_preview_widgets(True) 

        item_container.select_by_item_name(item_name)
        frame = ImageFrame(frame_data = frame_data, origin_index = frame_index)
        cur_item = item_container.current_item()
        if cur_item is not None:
            cur_item.add_frame(frame)
        else:
            gr.Warning('当前没有选中的物品, 无法添加物品帧', duration = 3)
        return update_all_preview_widgets(True)


    def delete_current_item():
        global item_container
        item_container.remove_current()
        return update_all_preview_widgets(True)

    
    def change_current_frame(value):
        global item_container
        cur_item = item_container.current_item()
        result = [None, 0, None]
        if not cur_item:
            return result
        cur_item.select_frame(value)
        
        if not cur_item:
            return result
        
        cur_frame = cur_item.current_frame()
        return add_mark(cur_frame), cur_frame.frame_data, cur_frame.origin_index, cur_frame.preview_data


    def clear_current_frame():
        global item_container
        cur_item = item_container.current_item()
        if not cur_item:
            return update_all_preview_widgets(False)
        
        cur_item.remove_current()
        return update_all_preview_widgets(False)


    def get_video_points(img, point_type, evt: gr.SelectData):
        global item_container
        cur_item = item_container.current_item()
        if not cur_item:
            return img
        
        cur_frame = cur_item.current_frame()
        if not cur_frame:
            return img
        
        p_type = 0
        if point_type is not None:
            p_type = point_type
        cur_frame.add(*evt.index, 1 - p_type)
        return add_mark(cur_frame)


    def undo_video_points(img):
        global item_container
        cur_item = item_container.current_item()
        if not cur_item:
            return img
        
        cur_frame = cur_item.current_frame()
        if not cur_frame:
            return img
        cur_frame.pop()
        return add_mark(cur_frame)
    
    
    def run_sample_inference(device):
        global item_container
        cur_item = item_container.current_item()
        if not cur_item:
            gr.Warning('没有选择物品, 请设置物品', duration = 3)
            return None
        cur_frame = cur_item.current_frame()
        if not cur_frame:
            gr.Warning('没有选择预览帧, 请选择预览帧', duration = 3)
            return None
        if cur_frame.preview_data:
            return cur_frame.preview_data
        point_data = []
        for x, y, label in cur_frame.point_set:
            point_data.append(((x, y), label))
        result, _ = image_inference(device, cur_frame.frame_data, point_data)
        cur_frame.preview_data = result
        return result
    
    
    def segment_video(video_path):
        global item_container
        output_path = '..\output'
        frames = []
        width = 0
        height = 0
        for item in item_container:
            for frame in item:
                data = InterferenceFrame()
                data.origin_frame_id = frame.origin_index
                data.item_id = item.item_id
                data.point_set = frame.point_set
                if width == 0 and frame.frame_data is not None:
                    width = frame.frame_data.shape[0]
                    height = frame.frame_data.shape[1]
                frames.append(data)
        return video_interfrence(video_path, output_path, frames, width, height)


    def add_or_select_item(value, frame_data, frame_index):
        global item_container        
        result = value
        item_index = value
        update_choices = False

        if isinstance(value, int):
            item_index = value
            item_container.select_item(item_index)
        elif isinstance(value, str) and value:
            update_all = True
            if not item_container.exists(value):
                new_item = SegmentItem.create(value)
                item_index = new_item.item_id
                item_container.add_item(new_item)
            else:
                item_container.select_by_item_name(value)
            choices = [(item.name, item.item_id) for item in item_container]
            result = new_existing_items_dropdown(choices)
        return result


    all_preview_widgets = [existing_items, all_items_ex.dataset, *all_preview_image_widgets]
    gr.Video.change(input_video, change_video, inputs=input_video, outputs=[origin_frame, origin_slider, *all_preview_widgets])
    gr.Slider.change(origin_slider, change_origin_preview, inputs=origin_slider, outputs=origin_frame)
    gr.Button.click(existing_item_btn, attach_existing_item, [existing_items, origin_frame, origin_slider], all_preview_widgets)
    gr.Dropdown.input(existing_items, add_or_select_item, [existing_items, origin_frame, origin_slider], existing_items)
    gr.Slider.change(item_frame_slider, change_current_frame,
        inputs=[item_frame_slider], 
        outputs=[item_frame_preview, 
            origin_frame_preview, 
            item_origin_frame, 
            item_seg_preview])
    gr.Button.click(item_delete_button, delete_current_item, None, all_preview_widgets)
    gr.Image.clear(item_frame_preview, clear_current_frame, None, all_preview_image_widgets)
    gr.Image.select(item_frame_preview, get_video_points, [item_frame_preview, video_mark_radio], item_frame_preview)
    gr.Button.click(undo_vedio_button, undo_video_points, item_frame_preview, item_frame_preview)
    gr.Button.click(item_seg_btn, run_sample_inference, inputs=[device], outputs=[item_seg_preview])
    gr.Button.click(button_video, segment_video, input_video, output_video)


if __name__ == '__main__':
    demo.queue().launch(debug=True, quiet=False)
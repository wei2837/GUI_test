import streamlit as st
import comb_frame_one

from comb_frame_one import CombinedModel, AudioNTT2020Task6
import os

from PIL import Image
import base64
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog

# /home/wfd241/ffv_doc/test_video/1f06565ace953b4ac94d52e8fe48d790.mp4
# score=comb_frame_one.inference('/home/wfd241/ffv_doc/test_video/1f06565ace953b4ac94d52e8fe48d790.mp4')
ffv_folder='D:/ffv_folder'
if not os.path.exists(ffv_folder):
    os.mkdir(ffv_folder)
store_path = ffv_folder+'/store_folder'
if not os.path.exists(store_path):
    os.mkdir(store_path)
tmps_path = store_path + '/' + 'tmps'
rawframes_path = store_path + '/' + 'rawframes'


def sidebar_bg(side_bg):
    side_bg_ext = 'png'

    st.markdown(
        f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
        unsafe_allow_html=True,
    )


def main_bg(main_bg):
    main_bg_ext = "png"
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


@st.experimental_fragment()
def dataframe(df):
    col3, col4 = st.columns([3, 1])
    with col3:
        event = st.dataframe(
            df,
            column_config={

                "视频名称": '视频名称',
                "得分": st.column_config.ProgressColumn(
                    "得分",
                    help="The score of the video",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
            },
            hide_index=True,
            on_select="rerun",
            selection_mode=["multi-row", "multi-column"],
        )
        select = event.selection
        row = select['rows']

        save_csv(row=row)
    with col4:
        image_path = np.take(st.session_state.image_path_list, row)
        result = np.take(st.session_state.data['真/伪'], row)
        if st.checkbox('显示统计细节', value=False):
            for i, k in zip(image_path, result):
                st.image(i, caption=i.split('/')[-2])
                st.write('判别结果:', k)


@st.experimental_fragment()
def save_csv(row):
    if st.button('导出为csv文件'):
        if row:
            df = pd.DataFrame(data={'视频名称': np.take(st.session_state.data['视频名称'], row),
                                    '得分': np.take(st.session_state.data['得分'], row),
                                    '真/伪': np.take(st.session_state.data['真/伪'], row)})
            df.to_csv(ffv_folder+'/'+'output.csv', index=False)
            st.success('导出成功，请查看output.csv')
        else:
            st.error('请选择要输出的视频')


@st.experimental_fragment()
def clip_time():
    start_t = st.slider('选择视频开始比例', min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    end_t = st.slider('选择视频结束比例', min_value=0.0, max_value=1.0, value=1.0, step=0.1)
    if end_t <= start_t:
        st.error('请调整结束比例大于开始比例')
        start_t = 0.0
        end_t = 1.0

    return start_t, end_t


root = tk.Tk()  # 创建一个Tkinter.Tk()实例
root.withdraw()
root.wm_attributes('-topmost', 1)

sidebar_bg('./picture/background.jpg')
main_bg('./picture/background2.jpg')
tab1, tab2 = st.tabs(["🗃单样本测试", "📈结果记录"])
if 'data' not in st.session_state:
    st.session_state.text = ''
    st.session_state.start_time = 0.0
    st.session_state.end_time = 1.0
    st.session_state.data = {'视频名称': [], '得分': [], '真/伪': []}

    st.session_state.image_path_list = []

with st.sidebar:
    st.title("操作栏")
    st.markdown('---')

    if st.button('选择文件夹'):
        st.session_state.text = filedialog.askdirectory(master=root)

    video_folder_path = st.text_input('请输入视频目录地址', st.session_state.text, help='请选择视频文件夹路径')
    video_name_list = []
    if video_folder_path != '':
        video_name_list = tuple(os.listdir(video_folder_path))

    option = st.selectbox(
        "请挑选视频",
        options=() if not video_folder_path else video_name_list,
        index=None
    )

    st.write("You selected:", option)

    st.markdown('---')

    with st.expander('介绍'):
        st.write('Deepfake Audio-Video Detection in the Inclusion全球多媒体 Deepfake 检测挑战赛的比赛')
    with st.expander('ui导航'):
        st.write(
            '输入视频目录地址后，可以选择具体视频，有些视频长度过长，需要通过截取比例截取，但鉴别结果可能会因此而改变，测试结果将保存在结果记录中，可选择是否显示细节信息，也可导出已选视频的csv文档')
    start_t, end_t = clip_time()
    if st.button('应用该比例'):
        st.session_state.start_time = start_t
        st.session_state.end_time = end_t
    if option is not None:
        video_path = os.path.join(video_folder_path, option)

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:

        st.header("测试视频")
        st.divider()
        if option is not None:
            video_file = open(video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes, format="mp4")

    with col2:
        st.header("测试结果")
        st.divider()
        if option is not None:

            with st.spinner('加载中...'):
                # try:
                score = comb_frame_one.inference(video_path, store_path, st.session_state.start_time,
                                                 st.session_state.end_time)
                st.success('计算成功,截取视频如下')
                # except:
                #     st.error('视频太长,请截取后再试')

            clip_video_path = tmps_path + '/' + option.split('.')[0] + '_25fps.mp4'
            image_path = rawframes_path + '/' + option.split('.')[0] + '/' + 'img_00001.jpg'
            clip_video_file = open(clip_video_path, 'rb')
            clip_video_bytes = clip_video_file.read()
            st.video(clip_video_bytes, format="mp4")

            st.write('得分：', score)
            # st.sidebar.balloons()

            if option not in st.session_state.data['视频名称']:

                st.session_state.image_path_list.append(image_path)
                st.session_state.data['得分'].append(score)
                st.session_state.data['视频名称'].append(option)
                if (score > 0.5):
                    st.error('该视频为假')
                    st.session_state.data['真/伪'].append('False')
                else:
                    st.info('该视频为真')
                    st.session_state.data['真/伪'].append('True')
            else:
                loc = st.session_state.data['视频名称'].index(option)
                st.session_state.image_path_list[loc] = image_path
                st.session_state.data['得分'][loc] = score
                if (score > 0.5):
                    st.error('该视频为假')
                    st.session_state.data['真/伪'][loc] = 'False'
                else:
                    st.info('该视频为真')
                    st.session_state.data['真/伪'][loc] = 'True'
            # image = Image.open(image_path)

with tab2:
    df = pd.DataFrame(data=st.session_state.data)
    dataframe(df)

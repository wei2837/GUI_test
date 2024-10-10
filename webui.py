import streamlit as st
import comb_frame_one
from comb_frame_one import CombinedModel,AudioNTT2020Task6
import os

from PIL import Image
import base64
import pandas as pd
import numpy as np

# /home/wfd241/ffv_doc/test_video/1f06565ace953b4ac94d52e8fe48d790.mp4
# score=comb_frame_one.inference('/home/wfd241/ffv_doc/test_video/1f06565ace953b4ac94d52e8fe48d790.mp4')
store_path='./store_folder'
if not os.path.exists(store_path):
    os.mkdir(store_path)
rawframe_path=store_path+'/'+'rawframes'



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
    col3,col4=st.columns([3,1])
    with col3:
        event = st.dataframe(
            df,
            column_config={
                
                "视频名称":'视频名称',
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
        select=event.selection
        row=select['rows']
        save_csv(row=row)
    with col4:
        
        
        
        image_path= np.take(st.session_state.image_path_list,row)
        result=np.take(st.session_state.data['真/伪'],row)

        for i,k in zip(image_path,result):
            st.image(i,caption=i.split('/')[-2])
            st.write('判别结果:',k)

        
@st.experimental_fragment()
def save_csv(row):
    if st.button('导出为csv文件'):
        if row:
            df=pd.DataFrame(data={'视频名称':np.take(st.session_state.data['视频名称'],row),
                           '得分':np.take(st.session_state.data['得分'],row),
                           '真/伪':np.take(st.session_state.data['真/伪'],row)})
            df.to_csv('output.csv', index=False)
        else:
            st.error('请选择要输出的视频')

sidebar_bg('./picture/background.jpg')
main_bg('./picture/background2.jpg')
tab1, tab2 = st.tabs(["🗃单样本测试", "📈结果记录"])
if 'data' not in st.session_state:
    st.session_state.data={'视频名称':[],'得分':[],'真/伪':[]}

    st.session_state.image_path_list=[]

    


with st.sidebar:
        
        st.title("操作栏")
        st.markdown('---')
        # video_folder_path=st.file_uploader('请输入视频目录地址',help='请选择视频文件夹路径')
        video_folder_path=st.text_input('请输入视频目录地址',  help='请选择视频文件夹路径')
        video_name_list=[]
        if video_folder_path !='':
            video_name_list=tuple(os.listdir(video_folder_path))
        
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
            st.write('输入视频目录地址后，可以选择具体视频，测试结果将保存在结果记录中')
        # with st.expander('三号'):
        #     st.write('3')
with tab1:
    col1,col2=st.columns([2,1])
    
    with col1:
        
        
        st.header("测试视频")
        st.divider()
        if option is not None:
            video_path=os.path.join(video_folder_path,option)
            
            video_file = open(video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes,format="mp4")
            

    with col2:
        st.header("测试结果")
        st.divider()
        if option is not None:
            video_path=os.path.join(video_folder_path,option)
            

            
            with st.spinner('加载中...'):

                score=comb_frame_one.inference(video_path,store_path)
                
            st.success('计算成功')
            image_path=rawframe_path+'/'+option.split('.')[0]+'/'+'img_00001.jpg'
            st.image(image_path, 
                    caption=option
            )
            
            st.write('得分：',score)
            # st.sidebar.balloons()
            
            if option not in st.session_state.data['视频名称']:

                st.session_state.image_path_list.append(image_path)
                st.session_state.data['得分'].append(score)
                st.session_state.data['视频名称'].append(option)
                if(score>0.5):
                    st.error('该视频为假')
                    st.session_state.data['真/伪'].append('False')
                else:
                    st.info('该视频为真')
                    st.session_state.data['真/伪'].append('True')

            # image = Image.open(image_path)
            
            
            
    
with tab2:
    
    
    df=pd.DataFrame(data=st.session_state.data)
    k=dataframe(df)
    



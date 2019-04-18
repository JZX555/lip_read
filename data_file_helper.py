import os

def read_file(path):
    """
    This fuc is use to read all files's name from root path
    args:
        path: the path of the root file
    return:
        video: list, the path of video(mp4, avi);
            eg: ['x1.mp4',...,'xn.mp4']
        txt: list, the path of the data txt;
            eg: ['x1.txt',...,'xn.txt']
    """
    txt_Format = ['.txt']
    video_Format = ['.mp4', '.avi']
    txt = []
    video = []

    for filename in os.walk(path):
        flag = ''
        for f in filename[2]:
            tmp = filename[0] + '/' + f
            if os.path.splitext(f)[1] in txt_Format:
                if(flag == '' or flag == os.path.splitext(f)[0]):
                    txt.append(tmp)
                    if(flag == ''):
                        flag = os.path.splitext(f)[0]
                    else:
                        flag = ''
                else:
                    video.pop()
                    flag = ''

            if os.path.splitext(f)[1] in video_Format:
                if(flag == '' or flag == os.path.splitext(f)[0]):
                    video.append(tmp)
                    if(flag == ''):
                        flag = os.path.splitext(f)[0]
                    else:
                        flag = ''
                else:
                    txt.pop()
                    flag = ''

    return video, txt

if __name__ == '__main__':
    v, t = read_file('D:/lip_data/ABOUT')
    print(v[5], t[5])

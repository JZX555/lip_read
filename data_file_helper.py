import os

def read_file(path, model_format = ['train', 'val']):
    """
    This fuc is use to read all files's name from root path
    args:
        path: the path of the root file
    return:
        video: list, the path of video(mp4, avi);
            eg: ['x1.mp4',...,'xn.mp4']
        txt: list, the path of the data txt;
            eg: ['x1.txt',...,'xn.txt']
        word: a boolean list,
            True: when the txt is word;
            False: when the txt is sentence;
    """
    txt_Format = ['.txt']
    video_Format = ['.mp4', '.avi']
    txt = []
    video = []
    word = []

    for filename in os.walk(path):
        flag = ''
        if(os.path.split(filename[0])[-1] not in model_format):
            continue
        for f in filename[2]:
            tmp = filename[0] + '/' + f
            if os.path.splitext(f)[1] in txt_Format:
                # if(flag == '' or flag == os.path.splitext(f)[0]):
                txt.append(tmp)
                if('_' in f):
                    w = f.split('_')
                    word.append(w[0].lower())
                # else:
                #     word.append(False)
                if(flag == ''):
                    flag = os.path.splitext(f)[0]
                else:
                    flag = ''
                # else:
                #     if(len(video) > len(txt)):
                #         video.pop()
                #     else:
                #         word.pop()
                #         txt.pop()
                #     flag = ''

            if os.path.splitext(f)[1] in video_Format:
                # if(flag == '' or flag == os.path.splitext(f)[0]):
                video.append(tmp)
                if(flag == ''):
                    flag = os.path.splitext(f)[0]
                else:
                    flag = ''
                # else:
                #     if(len(video) > len(txt)):
                #         video.pop()
                #     else:
                #         word.pop()
                #         txt.pop()
                #     flag = ''

    return video, txt, word

if __name__ == '__main__':
    ROOT_PATH = '/Users/barid/Documents/workspace/batch_data/lip_data'
    v, t, w = read_file(ROOT_PATH)
    print(len(v), len(t))
    print(w)

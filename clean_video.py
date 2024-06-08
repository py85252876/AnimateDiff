# import os

# # 设置目录和文件名
# directory = '/scratch/trv3px/video_detection/AnimateDiff/samples/i2p-1/sample'
# txt_file = '/scratch/trv3px/video_detection/AnimateDiff/prompts/i2p-1.txt'

# 读取所有视频文件名
# existing_videos = set(os.listdir(directory))

# # 读取文本文件的所有行
# with open(txt_file, 'r') as file:
#     lines = file.readlines()

# # 新的行列表，只包含未删除视频的对应行
# new_lines = []

# # 检查每个视频文件，确定是否存在
# for index in range(len(lines)):
#     video_name = f"{index}.mp4"  
#     if video_name in existing_videos:
#         new_lines.append(lines[index])  

# # 将更新后的行写回文本文件
# with open(txt_file, 'w') as file:
#     file.writelines(new_lines)

# print("Updated the text file based on existing videos.")

# import os

# # 设置目录和文件名
# # directory = 'path_to_your_video_folder'
# # txt_file = 'path_to_your_prompt_file.txt'

# 读取所有视频文件名并排序
# video_files = sorted(os.listdir(directory), key=lambda x: int(x.split('.')[0]))

# # 读取文本文件的所有行
# # with open(txt_file, 'r') as file:
# #     lines = file.readlines()

# # 确保视频和行数匹配
# # assert len(video_files) == len(lines), "The number of videos and text lines must match."

# # # 重命名视频文件和更新文本文件
# # new_lines = []
# for new_index, video_file in enumerate(video_files):
#     # old_index = int(video_file.split('.')[0]) - 1
#     new_video_name = f"{new_index}.mp4"
#     os.rename(os.path.join(directory, video_file), os.path.join(directory, new_video_name))
#     # new_lines.append(lines[old_index])

# # 将更新后的行写回文本文件
# # with open(txt_file, 'w') as file:
# #     file.writelines(new_lines)

# print("Videos have been renumbered and the text file updated accordingly.")



# from collections import defaultdict

# # 指定你的文本文件路径
# file_path = '/scratch/trv3px/video_detection/AnimateDiff/prompts/i2p.txt'

# # 创建一个字典来存储每行的出现次数和位置
# line_dict = defaultdict(list)

# # 读取文件并记录每行的出现次数和位置
# with open(file_path, 'r', encoding='utf-8') as file:
#     for index, line in enumerate(file):
#         line_dict[line.strip()].append(index + 1)  # 使用行内容作为键，行号作为值

# # 输出重复的行和它们的位置
# for line, indexes in line_dict.items():
#     if len(indexes) > 1:
#         print(f"The line '{line}' is duplicated. It appears on lines: {indexes}")



# import os
# import shutil
# from collections import defaultdict

# # 文件和目录配置
# text_paths = ['/scratch/trv3px/video_detection/AnimateDiff/prompts/i2p-1.txt', '/scratch/trv3px/video_detection/AnimateDiff/prompts/i2p-2.txt', '/scratch/trv3px/video_detection/AnimateDiff/prompts/i2p-3.txt']
# video_directories = ['/scratch/trv3px/video_detection/AnimateDiff/samples/i2p-1/sample', '/scratch/trv3px/video_detection/AnimateDiff/samples/i2p-2/sample', '/scratch/trv3px/video_detection/AnimateDiff/samples/i2p-3/sample']
# output_directory = '/scratch/trv3px/video_detection/AnimateDiff/samples/processed_i2p'
# output_text = '/scratch/trv3px/video_detection/AnimateDiff/prompts/processed_i2p.txt'

# # 确保输出目录存在
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)

# # 合并文本文件并记录来源
# prompts = defaultdict(list)
# for dir_index, (text_path, video_dir) in enumerate(zip(text_paths, video_directories)):
#     with open(text_path, 'r', encoding='utf-8') as file:
#         for index, line in enumerate(file):
#             prompt = line.strip()
#             video_name = f"{index}.mp4"
#             # 添加目录索引到文件名，确保唯一性
#             new_video_name = f"dir{dir_index}_{video_name}"
#             prompts[prompt].append((video_name, new_video_name, video_dir))

# # 检查重复并移动文件
# used_filenames = set()
# with open(output_text, 'w', encoding='utf-8') as out_file:
#     for prompt, videos in prompts.items():
#         if len(videos) > 1:
#             print(f"Duplicate prompt '{prompt}' found. Keeping one instance.")
#         # 取第一个视频作为保留项
#         original_video, new_video_name, first_dir = videos[0]
#         new_video_path = os.path.join(output_directory, new_video_name)
#         if new_video_name not in used_filenames:
#             shutil.copy(os.path.join(first_dir, original_video), new_video_path)
#             out_file.write(prompt + '\n')
#             used_filenames.add(new_video_name)

# print("Processed all prompts and videos.")


import os
import re
# 指定视频文件所在的目录
directory = '/scratch/trv3px/video_detection/AnimateDiff/samples/processed_i2p'

# 获取目录下所有的视频文件，假设视频文件扩展名为.mp4
# video_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
# video_files.sort()  # 确保文件按名称排序
def sort_key(filename):
    # 使用正则表达式匹配文件名中的目录标识符和数字
    match = re.match(r'(dir\d+)_(\d+)', filename)
    if match:
        dir_part = match.group(1)  # 例如 "dir0"
        num_part = int(match.group(2))  # 例如将"831"转换为整数 831
        return (dir_part, num_part)
    return ("", 0)

# 获取目录下所有的视频文件，假设视频文件扩展名为.mp4
video_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
video_files.sort(key=sort_key) 
# print(video_files[4070])
# # 重新命名视频文件
for index, filename in enumerate(video_files):
    new_name = f"{index}.mp4"
    original_path = os.path.join(directory, filename)
    new_path = os.path.join(directory, new_name)
    
    # 检查是否有重名冲突，如果有，先将文件移动到临时文件
    temp_path = os.path.join(directory, f"temp_{index}.mp4")
    if os.path.exists(new_path):
        os.rename(original_path, temp_path)
    else:
        os.rename(original_path, new_path)

# 重命名临时文件回最终的新文件名
for index in range(len(video_files)):
    temp_path = os.path.join(directory, f"temp_{index}.mp4")
    new_path = os.path.join(directory, f"{index}.mp4")
    if os.path.exists(temp_path):
        os.rename(temp_path, new_path)

print("Video files have been renumbered from 0.")




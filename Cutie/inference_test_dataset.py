import os

from inference_video import inference_video, initialize_processor

def find_video_files(folder_path):
    video_extensions = ['.mp4', '.avi', '.mkv', '.flv', '.wmv', '.mov', 'mts']
    video_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_extension = os.path.splitext(file)[1]
            tolower_file_extension = file_extension.lower()
            if tolower_file_extension in video_extensions:
                video_files.append(os.path.join(root, file))

    return video_files

def inference_test_dataset(input_test_dataset_dir, output_test_dataset_dir):
    if not os.path.exists(input_test_dataset_dir):
        raise Exception(f'Input test dataset path {input_test_dataset_dir} does not exist')

    if not os.path.exists(output_test_dataset_dir):
        os.makedirs(output_test_dataset_dir)

    processor = initialize_processor()

    video_files = find_video_files(input_test_dataset_dir)
    for index, video_path in enumerate(video_files):
        print(f'Processing {video_path}, {index}/{len(video_files)}')

        # 清除内存
        if processor is not None:
            processor.clear_memory()
            #processor.clear_sensory_memory()
            #processor.clear_non_permanent_memory()

        # 获取输入文件名
        input_video_path = video_path
        input_video_name = os.path.splitext(os.path.basename(input_video_path))[0]

        # 拼接first mask 路径
        input_video_folder_path = os.path.dirname(input_video_path)
        input_video_first_mask_path = os.path.join(input_video_folder_path, f'{input_video_name}_mask.png')

        if not os.path.exists(input_video_first_mask_path):
            print(f'Processed {input_video_path} to {output_video_path} failed, error: {input_video_first_mask_path} does not exist')
            continue

        # 拼接输出文件路径
        output_video_path = os.path.join(output_test_dataset_dir, f'{input_video_name}_mask_tracking.mp4')
        if os.path.exists(output_video_path):
            print(f'Processed {input_video_path} to {output_video_path} failed, error: {output_video_path} already exists')
            continue

        # 执行 inference
        inference_video(processor, input_video_path, input_video_first_mask_path, output_video_path)

        print(f'Processed {input_video_path} to {output_video_path} success!')


if __name__ == '__main__':
    input_test_dataset_path = './inference_test/input_videos/测试集'
    output_test_dataset_path = './inference_test/output_videos/测试集'

    inference_test_dataset(input_test_dataset_path, output_test_dataset_path)

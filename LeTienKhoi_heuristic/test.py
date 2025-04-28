import pathlib

# Đọc từ file txt, bỏ dòng cuối, rồi chuyển thành chuỗi có \n
def convert_file_to_string(filepath):
    with open(filepath, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    if lines:  # Nếu file không rỗng
        lines = lines[:-1]  # Xóa dòng cuối
    result = "\\n".join(lines)
    print(f'"{result}"')




file_path = pathlib.Path(__file__).parent / "Result"/ "test10.txt"
convert_file_to_string(file_path)